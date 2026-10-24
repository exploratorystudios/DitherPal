"""Background donor patches anchored to the source surface, prepared once per animation."""
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

    def render(self, width, height, focal, camera, yaw):
        if not len(self.faces):
            return (np.zeros((height, width, 3), np.uint8),
                    np.full((height, width), np.inf, np.float32), np.zeros((height, width), np.float32))
        return rasterize_alpha(self.vertices, self.uv, self.faces, self.texture,
                               width, height, focal, camera, yaw, self.alpha)


def _layer(vertices, uv, faces, alpha, texture):
    return RepairLayer(np.ascontiguousarray(vertices, np.float32).reshape(-1, 3),
                       np.ascontiguousarray(uv, np.float32).reshape(-1, 2),
                       np.ascontiguousarray(faces, np.int32).reshape(-1, 3),
                       np.ascontiguousarray(alpha, np.float32), texture)


def _strips(mesh, edges, kind, texture, depth_map, cancelled):
    """Fallback background overlap and the tested narrow foreground feather."""
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


def _donors(mesh, texture, depth_map, edges, cancelled, progress):
    dims = np.array([mesh.width-1, mesh.height-1], np.float32)
    density = max(1., (texture.shape[1]-1)/dims[0], (texture.shape[0]-1)/dims[1])
    patches, rejected = [], 0
    for index, (kind, p, q, inside) in enumerate(edges):
        _check_cancelled(cancelled)
        if index % 128 == 0 and progress:
            progress(f'Selecting background donor patches ({index}/{len(edges)})…')
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
        patches.append(dict(points=points, tangent=tangent, normal=normal, length=float(length),
                            copied=copied, depth=mesh.vertices[[p, q], 2], offset=offset, shift=shift))
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
        texture[y-pad:y+h+pad, x-pad:x+w+pad] = cv2.copyMakeBorder(patch['copied'], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        offset = len(vertices)
        for normal_distance, opacity in [(-OVERLAP, 0.), (0., 1.), (EXTENSION, 1.)]:
            for along in [-1., patch['length']+1.]:
                point = patch['points'][0]+patch['tangent']*along+patch['normal']*normal_distance
                z = np.interp(along, [0, patch['length']], patch['depth'])
                vertices.append([*((point-dims/2)*z/mesh.focal), z])
                tx = x+(along+1)/(patch['length']+2)*(w-1)
                ty = y+(normal_distance+OVERLAP)/(EXTENSION+OVERLAP)*(h-1)
                uv.append([tx/(atlas_width-1), ty/(atlas_height-1)])
                alpha.append(opacity)
        for row in range(2):
            a = offset+row*2
            faces.extend([(a, a+2, a+1), (a+1, a+2, a+3)])
    return _layer(vertices, uv, faces, alpha, texture)


def _blend(base, color, opacity):
    return np.round(base.astype(np.float32)*(1-opacity[..., None])
                    + color.astype(np.float32)*opacity[..., None]).astype(np.uint8)


class DonorRepair:
    """The selected donor-copy pipeline, with overlap/fill fallback and a 2px foreground feather."""

    def __init__(self, mesh, texture, depth_map, cancelled=lambda: False, progress=None):
        _check_cancelled(cancelled)
        edges = _boundaries(mesh)
        self.background = _strips(mesh, edges, 'background', texture, depth_map, cancelled)
        self.foreground = _strips(mesh, edges, 'foreground', texture, depth_map, cancelled)
        patches, self.skipped = _donors(mesh, texture, depth_map, edges, cancelled, progress)
        self.donors = _atlas(mesh, patches)
        self.locations = [dict(points=p['points'].tolist(), normal=p['normal'].tolist(),
                               tangent=p['tangent'].tolist(), offset=p['offset'], shift=p['shift']) for p in patches]
        _check_cancelled(cancelled)

    def summary(self):
        return dict(method='background_donor_patches', patches=len(self.locations), skipped_edges=self.skipped,
                    extension_source_pixels=EXTENSION, overlap_source_pixels=OVERLAP,
                    foreground_feather_source_pixels=FOREGROUND_FEATHER,
                    fallback='Background overlap, then Telea filling of residual gaps')

    def write(self, folder):
        folder = Path(folder)
        if not any(len(layer.faces) for layer in [self.background, self.foreground, self.donors]):
            return
        arrays = {}
        for name, layer in [('background', self.background), ('foreground', self.foreground), ('donors', self.donors)]:
            for key in ['vertices', 'uv', 'faces', 'alpha']:
                arrays[f'{name}_{key}'] = getattr(layer, key)
        np.savez_compressed(folder/'repair_layers.npz', **arrays)
        if len(self.donors.faces):
            Image.fromarray(self.donors.texture).save(folder/'donor_texture.png')
        (folder/'donor_locations.json').write_text(json.dumps(self.locations, indent=2)+'\n')

    def render(self, raw, depth, focal, camera, yaw=0.):
        height, width = raw.shape[:2]
        missing = ~np.isfinite(depth)
        background, background_z, _ = self.background.render(width, height, focal, camera, yaw)
        supplied = missing & np.isfinite(background_z)
        base = raw.copy()
        base[supplied] = background[supplied]
        remaining = missing & ~supplied
        if remaining.any() and not remaining.all():
            base = cv2.inpaint(base, np.uint8(remaining)*255, 3, cv2.INPAINT_TELEA)
        combined_depth = depth.copy()
        combined_depth[supplied] = background_z[supplied]
        color, patch_z, alpha = self.donors.render(width, height, focal, camera, yaw)
        same_background = np.zeros_like(missing)
        known = ~missing & np.isfinite(patch_z)
        same_background[known] = np.maximum(patch_z[known]/depth[known], depth[known]/patch_z[known]) < 1.03
        visible = np.isfinite(patch_z) & (missing | same_background)
        opacity = np.where(visible, np.clip(alpha, 0, 1), 0)
        repaired = _blend(base, color, opacity)
        combined_depth[missing & visible] = patch_z[missing & visible]
        foreground, foreground_z, foreground_alpha = self.foreground.render(width, height, focal, camera, yaw)
        feather = np.where(foreground_z < combined_depth-1e-5, np.clip(foreground_alpha, 0, 1), 0)
        repaired = _blend(repaired, foreground, feather)
        return repaired, combined_depth, opacity, feather
