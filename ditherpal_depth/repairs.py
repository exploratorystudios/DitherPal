"""Background-only seams: stable geometry, per-frame healing, and optional pixel art."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .renderer import rasterize_alpha

EXTENSION = 12.
OVERLAP = 3.
FOREGROUND_FEATHER = 2.
CONTINUATION = 28.


def _never():
    return False


def _check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError()


def _boundaries(mesh):
    """Recover near/far edges of the regular depth grid beside omitted faces."""
    cols = len(np.unique(mesh.uv[:, 0]))
    rows = len(mesh.vertices)//cols
    indices = np.arange(rows*cols).reshape(rows, cols)
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    full = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3),
                           np.stack((b, c, d), -1).reshape(-1, 3)))
    count = len(mesh.vertices)

    def face_keys(faces):
        sorted_faces = np.sort(faces.astype(np.int64), axis=1)
        return (sorted_faces[:, 0]*count+sorted_faces[:, 1])*count+sorted_faces[:, 2]

    kept = np.isin(face_keys(full), face_keys(mesh.faces), assume_unique=True)
    # Arrays avoid retaining a Python adjacency object for every mesh edge.
    edges = full[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    opposite = full[:, [2, 0, 1]].reshape(-1)
    sorted_edges = np.sort(edges, axis=1)
    keys = sorted_edges[:, 0].astype(np.int64)*count+sorted_edges[:, 1]
    _, occurrences = np.unique(keys, return_counts=True)
    order = np.argsort(keys, kind='stable')
    starts = np.concatenate(([0], np.cumsum(occurrences)[:-1]))
    paired = starts[occurrences == 2]
    first, second = order[paired], order[paired+1]
    boundary = kept[first//3] != kept[second//3]
    pairs = sorted(zip(first[boundary], second[boundary]), key=lambda pair: pair[0])
    result = []
    z = mesh.vertices[:, 2]
    for first, second in pairs:
        p, q = sorted_edges[first]
        inside, outside = (opposite[first], opposite[second]) if kept[first//3] else (opposite[second], opposite[first])
        edge_z = (z[p]+z[q])/2
        if z[outside] > edge_z*1.12:
            kind = 'foreground'
        elif edge_z > z[outside]*1.12:
            kind = 'background'
        else:
            continue
        result.append((kind, p, q, inside))
    return result


def _sample(image, xy, mesh):
    """Source-pixel coordinates also work with a larger baked output texture."""
    scale = np.array([(image.shape[1]-1)/(mesh.width-1), (image.shape[0]-1)/(mesh.height-1)], np.float32)
    coordinates = xy*scale
    return cv2.remap(image, np.ascontiguousarray(coordinates[..., 0], np.float32),
                     np.ascontiguousarray(coordinates[..., 1], np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


@dataclass
class RepairLayer:
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    alpha: np.ndarray
    texture: np.ndarray

    def render(self, width, height, focal, camera, yaw, min_depth=None):
        if not len(self.faces):
            return (np.zeros((height, width, 3), np.uint8),
                    np.full((height, width), np.inf, np.float32), np.zeros((height, width), np.float32))
        return rasterize_alpha(self.vertices, self.uv, self.faces, self.texture,
                               width, height, focal, camera, yaw, self.alpha, min_depth)


def _layer(vertices, uv, faces, alpha, texture):
    return RepairLayer(np.ascontiguousarray(vertices, np.float32).reshape(-1, 3),
                       np.ascontiguousarray(uv, np.float32).reshape(-1, 2),
                       np.ascontiguousarray(faces, np.int32).reshape(-1, 3),
                       np.ascontiguousarray(alpha, np.float32), texture)


def _strips(mesh, edges, kind, texture, depth_map, cancelled):
    """Source-anchored edge strips; the repair pipeline uses background edges only."""
    vertices, uv, alpha, faces = [], [], [], []
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    width = EXTENSION if kind == 'background' else FOREGROUND_FEATHER
    for index, (category, p, q, inside) in enumerate(edges):
        if index % 64 == 0:
            _check_cancelled(cancelled)
        if category != kind:
            continue
        points = mesh.uv[[p, q]]*dims
        tangent = points[1]-points[0]
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        normal /= np.linalg.norm(normal)
        if np.dot(normal, points.mean(0)-mesh.uv[inside]*dims) < 0:
            normal *= -1
        tex_uv = mesh.uv[[p, q]]*.75+mesh.uv[inside]*.25
        if kind == 'background':
            candidate = None
            for inset in [3., 5., 8.]:
                sample = np.clip((points-normal*inset)/dims, 0, 1)
                sx = np.round(sample[:, 0]*(depth_map.shape[1]-1)).astype(int)
                sy = np.round(sample[:, 1]*(depth_map.shape[0]-1)).astype(int)
                sampled_z = depth_map[sy, sx]
                edge_z = mesh.vertices[[p, q], 2]
                if (np.maximum(sampled_z/edge_z, edge_z/sampled_z) < 1.12).all():
                    candidate = sample
            if candidate is None:
                continue
            tex_uv = candidate
            t = np.linspace(0, 1, 9)[:, None]
            probes = tex_uv[0]*(1-t)+tex_uv[1]*t
            px = np.clip(np.round(probes[:, 0]*(texture.shape[1]-1)).astype(int), 0, texture.shape[1]-1)
            py = np.clip(np.round(probes[:, 1]*(texture.shape[0]-1)).astype(int), 0, texture.shape[0]-1)
            if np.ptp(texture[py, px].astype(np.int16), axis=0).max() > 40:
                continue
        xy = np.concatenate((points, points+normal*width))
        depth = mesh.vertices[[p, q, p, q], 2]
        xyz = np.column_stack(((xy-dims/2)*depth[:, None]/mesh.focal, depth))
        offset = len(vertices)
        vertices.extend(xyz)
        uv.extend(np.concatenate((tex_uv, tex_uv)))
        alpha.extend([1., 1., 0., 0.] if kind == 'foreground' else [1., 1., 1., 1.])
        faces.extend([(offset, offset+2, offset+1), (offset+1, offset+2, offset+3)])
    return _layer(vertices, uv, faces, alpha, texture)


def _heal_patch(copied, original, safe_overlap, across):
    """Match known background tone, then reconcile the donor's boundary gradients."""
    delta = original.astype(np.float32)-copied.astype(np.float32)
    weights = safe_overlap.astype(np.float32)
    correction = (delta*weights[..., None]).sum(0)/np.maximum(weights.sum(0)[:, None], 1)
    correction = cv2.GaussianBlur(correction[None], (3, 1), 0)[0]
    correction = np.clip(correction, -48, 48)
    fade = np.exp(-np.maximum(across, 0)/10)[:, None, None]
    target = np.clip(copied.astype(np.float32)+correction[None]*fade, 0, 255).round().astype(np.uint8)
    mask = np.full(copied.shape[:2], 255, np.uint8)
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    height, width = copied.shape[:2]
    return cv2.seamlessClone(copied, target, mask, (width//2, height//2), cv2.NORMAL_CLONE)


def _donors(mesh, texture, depth_map, edges, cancelled, progress):
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    density = max(1., (texture.shape[1]-1)/dims[0], (texture.shape[0]-1)/dims[1])
    patches, rejected = [], 0
    for index, (kind, p, q, inside) in enumerate(edges):
        _check_cancelled(cancelled)
        if index % 128 == 0 and progress:
            progress(f'Selecting and healing background donor patches ({index}/{len(edges)})…')
        if kind != 'background':
            continue
        points = mesh.uv[[p, q]]*dims
        length = np.linalg.norm(points[1]-points[0])
        tangent = (points[1]-points[0])/length
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        if np.dot(normal, points.mean(0)-mesh.uv[inside]*dims) < 0:
            normal *= -1
        along = np.linspace(-1., length+1., max(5, int(np.ceil((length+2)*density))+1), dtype=np.float32)
        across = np.linspace(-OVERLAP, EXTENSION, int(np.ceil((EXTENSION+OVERLAP)*density))+1, dtype=np.float32)
        ss, nn = np.meshgrid(along, across)
        destination = points[0]+ss[..., None]*tangent+nn[..., None]*normal
        expected_z = np.interp(along, [0, length], mesh.vertices[[p, q], 2])[None, :]
        original = _sample(texture, destination, mesh)
        original_depth = _sample(depth_map, destination, mesh)
        safe_overlap = (nn < -.5) & (np.maximum(original_depth/expected_z, expected_z/original_depth) < 1.15)
        if safe_overlap.sum() < len(along):
            rejected += 1
            continue
        best = None
        for offset in [15., 18., 23., 30.]:
            for shift in [0., -5., 5., -10., 10.]:
                donor_xy = destination-normal*offset+tangent*shift
                if not ((donor_xy[..., 0] >= 0) & (donor_xy[..., 0] <= dims[0])
                        & (donor_xy[..., 1] >= 0) & (donor_xy[..., 1] <= dims[1])).all():
                    continue
                z = _sample(depth_map, donor_xy, mesh)
                if not (np.maximum(z/expected_z, expected_z/z) < 1.18).all():
                    continue
                color = _sample(texture, donor_xy, mesh)
                diff = color.astype(np.float32)-original.astype(np.float32)
                cost = float(np.mean(diff[safe_overlap]**2))+offset*.5+abs(shift)*2
                if best is None or cost < best[0]:
                    best = cost, color, offset, shift
        if best is None:
            rejected += 1
            continue
        _, copied, offset, shift = best
        _check_cancelled(cancelled)
        healed = _heal_patch(copied, original, safe_overlap, across)
        _check_cancelled(cancelled)
        patches.append(dict(points=points, tangent=tangent, normal=normal, length=float(length),
                            copied=copied, healed=healed, depth=mesh.vertices[[p, q], 2], offset=offset, shift=shift,
                            density=density, extension=EXTENSION))
    return patches, rejected


def _atlas(mesh, patches):
    pad = 2
    atlas_width = max(1024, max((p['copied'].shape[1]+2*pad for p in patches), default=0))
    cursor_x, cursor_y, row_height = pad, pad, 0
    placements = []
    for patch in patches:
        h, w = patch['copied'].shape[:2]
        if cursor_x+w+pad > atlas_width:
            cursor_x, cursor_y, row_height = pad, cursor_y+row_height+2*pad, 0
        placements.append((cursor_x, cursor_y))
        cursor_x += w+2*pad
        row_height = max(row_height, h)
    atlas_height = max(8, cursor_y+row_height+pad)
    texture = np.zeros((atlas_height, atlas_width, 3), np.uint8)
    vertices, uv, alpha, faces = [], [], [], []
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    for patch, (x, y) in zip(patches, placements):
        h, w = patch['copied'].shape[:2]
        texture[y-pad:y+h+pad, x-pad:x+w+pad] = cv2.copyMakeBorder(patch['healed'], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        offset = len(vertices)
        extension = patch.get('extension', EXTENSION)
        for normal_distance, opacity in [(-OVERLAP, 0.), (0., 1.), (extension, 1.)]:
            for along in [-1., patch['length']+1.]:
                point = patch['points'][0]+patch['tangent']*along+patch['normal']*normal_distance
                z = np.interp(along, [0, patch['length']], patch['depth'])
                vertices.append([*((point-dims/2)*z/mesh.focal), z])
                tx = x+(along+1)/(patch['length']+2)*(w-1)
                ty = y+(normal_distance+OVERLAP)/(extension+OVERLAP)*(h-1)
                uv.append([tx/(atlas_width-1), ty/(atlas_height-1)])
                alpha.append(opacity)
        for row in range(2):
            a = offset+row*2
            faces.extend([(a, a+2, a+1), (a+1, a+2, a+3)])
    return _layer(vertices, uv, faces, alpha, texture)


def _blend(base, color, opacity):
    return np.round(base.astype(np.float32)*(1-opacity[..., None])
                    + color.astype(np.float32)*opacity[..., None]).astype(np.uint8)


def _patch_metadata(mesh, patches, atlas_width):
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    x, y, row_height = 2, 2, 0
    result = []
    for p in patches:
        h, w = p['copied'].shape[:2]
        if x+w+2 > atlas_width:
            x, y, row_height = 2, y+row_height+4, 0
        along = np.linspace(-1., p['length']+1., w, dtype=np.float32)
        across = np.linspace(-OVERLAP, p.get('extension', EXTENSION), h, dtype=np.float32)
        ss, nn = np.meshgrid(along, across)
        xy = p['points'][0]+ss[..., None]*p['tangent']+nn[..., None]*p['normal']
        z = np.broadcast_to(np.interp(along, [0, p['length']], p['depth'])[None], (h, w))
        result.append(dict(patch=p, world_xy=(xy-dims/2)*z[..., None]/mesh.focal,
                           z=z, across=across, overlap=nn < -.5, slot=(x, y)))
        x += w+4
        row_height = max(row_height, h)
    return result


def _clipped_continuation(mesh, metadata, depth, texture, cancelled):
    vertices, uv, faces, alpha = [], [], [], []
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    for item in metadata:
        _check_cancelled(cancelled)
        z = item['z']
        xy = item['world_xy']*mesh.focal/z[..., None]+dims/2
        x, y = np.rint(xy[..., 0]).astype(int), np.rint(xy[..., 1]).astype(int)
        valid = (x >= 0) & (x < mesh.width) & (y >= 0) & (y < mesh.height)
        source_z = depth[np.clip(y, 0, mesh.height-1), np.clip(x, 0, mesh.width-1)]
        valid &= source_z <= z*1.03
        h, w = z.shape
        start = len(vertices)
        vertices.extend(np.concatenate([item['world_xy'], z[..., None]], axis=2).reshape(-1, 3))
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        x0, y0 = item['slot']
        uv.extend(np.stack([(xx+x0)/(texture.shape[1]-1), (yy+y0)/(texture.shape[0]-1)], -1).reshape(-1, 2))
        opacity = np.broadcast_to(np.clip((item['across']+OVERLAP)/OVERLAP, 0, 1)[:, None], (h, w))
        alpha.extend(opacity.ravel())
        for row in range(h-1):
            if row % 32 == 0:
                _check_cancelled(cancelled)
            for col in range(w-1):
                # Omit unsupported faces, so transparent nearer geometry cannot
                # occlude a valid farther repair in the layer's depth buffer.
                if valid[row:row+2, col:col+2].all():
                    a = start+row*w+col
                    faces.extend([(a, a+w, a+1), (a+1, a+w, a+w+1)])
    return _layer(vertices, uv, faces, alpha, texture)


def _frame_atlas(raw, depth, camera, focal, yaw, metadata, fixed, cancelled):
    atlas = fixed.copy()
    height, width = depth.shape
    cosine, sine = np.cos(yaw), np.sin(yaw)
    for item in metadata:
        _check_cancelled(cancelled)
        p = item['patch']
        dx = item['world_xy'][..., 0]-camera[0]
        dy = item['world_xy'][..., 1]-camera[1]
        dz = item['z']-camera[2]
        px, z = cosine*dx-sine*dz, sine*dx+cosine*dz
        safe_z = np.maximum(z, 1e-9)
        x = np.rint(focal*px/safe_z+(width-1)/2).astype(int)
        y = np.rint(focal*dy/safe_z+(height-1)/2).astype(int)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0)
        cx, cy = np.clip(x, 0, width-1), np.clip(y, 0, height-1)
        observed = depth[cy, cx]
        safe = inside & item['overlap'] & np.isfinite(observed)
        safe &= np.maximum(observed/safe_z, safe_z/observed) < 1.15
        if safe.sum() < p['copied'].shape[1]:
            continue
        healed = _heal_patch(p['copied'], raw[cy, cx], safe, item['across'])
        _check_cancelled(cancelled)
        x, y = item['slot']
        h, w = healed.shape[:2]
        atlas[y-2:y+h+2, x-2:x+w+2] = cv2.copyMakeBorder(healed, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    return atlas


def _gap_background_limit(depth, radius=8):
    known = np.isfinite(depth)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    farther = cv2.dilate(np.where(known, depth, 0).astype(np.float32), kernel)
    nearer = cv2.erode(np.where(known, depth, np.inf).astype(np.float32), kernel)
    boundary = (farther > 0) & np.isfinite(nearer) & (farther/nearer > 1.12)
    limit = np.where(~known & boundary, farther/1.08, 0).astype(np.float32)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(~known), 8)
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        component = labels[y:y+h, x:x+w] == i
        region = limit[y:y+h, x:x+w]
        seeds = component & (region > 0)
        if seeds.any():
            _, nearest = cv2.distanceTransformWithLabels(np.uint8(~seeds), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
            values = region[seeds]
            fill = values[np.clip(nearest-1, 0, len(values)-1)]
            region[component & (region == 0)] = fill[component & (region == 0)]
    return limit


class DonorRepair:
    """Fill gaps exclusively with healed background geometry; preserve every visible source pixel."""

    def __init__(self, mesh, texture, depth_map, cancelled=lambda: False, progress=None, pixel_art=None):
        _check_cancelled(cancelled)
        self.mesh, self.cancelled, self.pixel_art = mesh, cancelled, pixel_art
        self.render_count = 0
        edges = [edge for edge in _boundaries(mesh) if edge[0] == 'background']
        self.background = _strips(mesh, edges, 'background', texture, depth_map, cancelled)
        # Retain an empty export entry for older repair-layer readers.
        self.foreground = _layer([], [], [], [], texture)
        patches, self.skipped = _donors(mesh, texture, depth_map, edges, cancelled, progress)
        self.donors = _atlas(mesh, patches)
        self._fixed = self.donors.texture
        self.metadata = _patch_metadata(mesh, patches, self._fixed.shape[1])
        if pixel_art is not None:
            pixel_art.prepare(mesh, texture, depth_map, self.metadata, cancelled)
        expanded = []
        for p, item in zip(patches, self.metadata):
            _check_cancelled(cancelled)
            q = dict(p)
            height = int(np.ceil((CONTINUATION+OVERLAP)*p['density']))+1
            extra = height-p['copied'].shape[0]
            q['copied'] = cv2.copyMakeBorder(p['copied'], 0, extra, 0, 0, cv2.BORDER_REFLECT_101)
            q['healed'] = cv2.copyMakeBorder(p['healed'], 0, extra, 0, 0, cv2.BORDER_REFLECT_101)
            q['extension'] = CONTINUATION
            if pixel_art is not None and item['art_context'] is not None:
                q['allowed_background_palette'] = item['art_context']['palette']
            expanded.append(q)
        expanded_atlas = _atlas(mesh, expanded)
        self._extended_fixed = expanded_atlas.texture
        self.extended_metadata = _patch_metadata(mesh, expanded, self._extended_fixed.shape[1])
        if pixel_art is not None:
            pixel_art.prepare(mesh, texture, depth_map, self.extended_metadata, cancelled)
        self.continuation = _clipped_continuation(mesh, self.extended_metadata, depth_map,
                                                  self._extended_fixed, cancelled)
        self.locations = [dict(points=p['points'].tolist(), normal=p['normal'].tolist(),
                               tangent=p['tangent'].tolist(), offset=p['offset'], shift=p['shift']) for p in patches]
        _check_cancelled(cancelled)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['cancelled'] = _never
        return state

    def summary(self):
        return dict(method='background_only_healed_donor_patches', patches=len(self.locations), skipped_edges=self.skipped,
                    healing='Per-frame background tone matching and OpenCV NORMAL_CLONE',
                    pixel_art=self.pixel_art.style if self.pixel_art is not None else 'off',
                    extension_source_pixels=CONTINUATION, overlap_source_pixels=OVERLAP,
                    foreground_seam_triangles=0, foreground_feather_source_pixels=0,
                    fallback='Copy accepted background repair pixels only')

    def write(self, folder):
        folder = Path(folder)
        layers = [('background', self.background), ('foreground', self.foreground),
                  ('donors', self.donors), ('continuation', self.continuation)]
        if not any(len(layer.faces) for _, layer in layers):
            return
        arrays = {}
        for name, layer in layers:
            for key in ['vertices', 'uv', 'faces', 'alpha']:
                arrays[f'{name}_{key}'] = getattr(layer, key)
        np.savez_compressed(folder/'repair_layers.npz', **arrays)
        if len(self.donors.faces):
            Image.fromarray(self.donors.texture).save(folder/'donor_texture.png')
            Image.fromarray(self.continuation.texture).save(folder/'continuation_texture.png')
        (folder/'donor_locations.json').write_text(json.dumps(self.locations, indent=2)+'\n')

    def write_peak(self, folder):
        if len(self.donors.faces):
            Image.fromarray(self.donors.texture).save(Path(folder)/'peak_donor_texture.png')
            Image.fromarray(self.continuation.texture).save(Path(folder)/'peak_continuation_texture.png')

    def render(self, raw, depth, focal, camera, yaw=0.):
        cancelled = self.cancelled
        _check_cancelled(cancelled)
        height, width = raw.shape[:2]
        missing = ~np.isfinite(depth)
        result, repaired_depth = raw.copy(), depth.copy()
        supplied = np.zeros(missing.shape, bool)
        zero = np.zeros(missing.shape, np.float32)
        if not missing.any():
            return result, repaired_depth, zero, zero.copy()
        if self.metadata:
            atlas = _frame_atlas(raw, depth, camera, focal, yaw, self.metadata, self._fixed, cancelled)
            extended = _frame_atlas(raw, depth, camera, focal, yaw, self.extended_metadata,
                                     self._extended_fixed, cancelled)
            if self.pixel_art is not None:
                atlas = self.pixel_art.dither(atlas, self.metadata, cancelled)
                extended = self.pixel_art.dither(extended, self.extended_metadata, cancelled)
            self.donors.texture, self.continuation.texture = atlas, extended
        limit = _gap_background_limit(depth, radius=max(1, round(8*width/self.mesh.width)))
        for layer in (self.background, self.continuation, self.donors):
            _check_cancelled(cancelled)
            color, z, alpha = layer.render(width, height, focal, camera, yaw, limit)
            take = missing & np.isfinite(z) & (alpha > .01)
            result[take], repaired_depth[take] = color[take], z[take]
            supplied[take] = True
        remaining = missing & ~supplied
        if remaining.any() and supplied.any():
            _, nearest = cv2.distanceTransformWithLabels(np.uint8(~supplied), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
            indices = np.clip(nearest-1, 0, int(supplied.sum())-1)
            colors, depths = result[supplied], repaired_depth[supplied]
            result[remaining] = colors[indices[remaining]]
            repaired_depth[remaining] = depths[indices[remaining]]
        self.render_count += 1
        _check_cancelled(cancelled)
        return result, repaired_depth, supplied.astype(np.float32), zero
