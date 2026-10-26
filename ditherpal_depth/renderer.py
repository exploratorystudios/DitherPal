"""Perspective-correct textured triangle rasterization with an actual Z buffer."""
import cv2
import numpy as np
from numba import njit


@njit(cache=True)
def _rasterize(vertices, uv, faces, texture, width, height, focal, camera, yaw, alpha, min_depth=None):
    image = np.zeros((height, width, 3), np.uint8)
    zbuffer = np.full((height, width), np.inf, np.float32)
    opacity = np.zeros((height, width) if alpha is not None else (0, 0), np.float32)
    projected = np.empty_like(vertices)
    clip_near = max(1e-9, np.min(vertices[:, 2]) * .01)
    c, s = np.cos(yaw), np.sin(yaw)
    for i in range(len(vertices)):
        x, y, z = vertices[i, 0] - camera[0], vertices[i, 1] - camera[1], vertices[i, 2] - camera[2]
        px, pz = c * x - s * z, s * x + c * z
        projected[i, 2] = pz
        if pz > clip_near:
            projected[i, 0] = focal * px / pz + (width - 1) / 2
            projected[i, 1] = focal * y / pz + (height - 1) / 2
    th, tw = texture.shape[:2]
    for face in faces:
        i0, i1, i2 = face
        x0, y0, z0 = projected[i0]
        x1, y1, z1 = projected[i1]
        x2, y2, z2 = projected[i2]
        if min(z0, z1, z2) <= clip_near:
            continue
        denominator = (y1-y2) * (x0-x2) + (x2-x1) * (y0-y2)
        if abs(denominator) < 1e-8:
            continue
        left, right = max(0, int(np.floor(min(x0,x1,x2)))), min(width-1, int(np.ceil(max(x0,x1,x2))))
        top, bottom = max(0, int(np.floor(min(y0,y1,y2)))), min(height-1, int(np.ceil(max(y0,y1,y2))))
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                a = ((y1-y2) * (x-x2) + (x2-x1) * (y-y2)) / denominator
                b = ((y2-y0) * (x-x2) + (x0-x2) * (y-y2)) / denominator
                d = 1-a-b
                if min(a,b,d) < -1e-5:
                    continue
                inverse_z = a/z0 + b/z1 + d/z2
                z = 1 / inverse_z
                if min_depth is not None and z < min_depth[y, x]:
                    continue
                if z >= zbuffer[y, x]:
                    continue
                u = (a*uv[i0,0]/z0 + b*uv[i1,0]/z1 + d*uv[i2,0]/z2) * z
                v = (a*uv[i0,1]/z0 + b*uv[i1,1]/z1 + d*uv[i2,1]/z2) * z
                tx, ty = max(0., min(tw-1., u*(tw-1))), max(0., min(th-1., v*(th-1)))
                ix, iy = int(tx), int(ty)
                jx, jy = min(ix+1, tw-1), min(iy+1, th-1)
                dx, dy = tx-ix, ty-iy
                for channel in range(3):
                    value = ((1-dy) * ((1-dx)*texture[iy,ix,channel] + dx*texture[iy,jx,channel])
                             + dy * ((1-dx)*texture[jy,ix,channel] + dx*texture[jy,jx,channel]))
                    image[y,x,channel] = np.uint8(max(0., min(255., value + 0.5)))
                zbuffer[y,x] = z
                if alpha is not None:
                    opacity[y,x] = (a*alpha[i0]/z0 + b*alpha[i1]/z1 + d*alpha[i2]/z2) * z
    return image, zbuffer, opacity


@njit(cache=True)
def rasterize(vertices, uv, faces, texture, width, height, focal, camera, yaw):
    image, zbuffer, _ = _rasterize(vertices, uv, faces, texture, width, height, focal, camera, yaw, None)
    return image, zbuffer


@njit(cache=True)
def rasterize_alpha(vertices, uv, faces, texture, width, height, focal, camera, yaw, alpha, min_depth=None):
    """Render a separate layer with perspective-correct per-vertex opacity."""
    return _rasterize(vertices, uv, faces, texture, width, height, focal, camera, yaw, alpha, min_depth)


def render(mesh, texture, camera=(0., 0., 0.), yaw=0., fill_holes=False):
    raw, zbuffer = rasterize(mesh.vertices, mesh.uv, mesh.faces, texture, mesh.width, mesh.height,
                            mesh.focal, np.array(camera, np.float32), float(yaw))
    missing = ~np.isfinite(zbuffer)
    if fill_holes and missing.any() and not missing.all():
        image = cv2.inpaint(raw, missing.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    else:
        image = raw
    return image, missing, raw


def ordered_dither(rgb, levels):
    """Optional post-render effect with a fixed RGB cube, independent of geometry."""
    if not levels:
        return rgb
    matrix = np.array([[0,8,2,10], [12,4,14,6], [3,11,1,9], [15,7,13,5]], np.float32)
    y, x = np.indices(rgb.shape[:2])
    threshold = ((matrix + 0.5) / 16 - 0.5)[y % 4, x % 4]
    step = 255 / (levels - 1)
    indices = np.clip(np.floor(rgb.astype(np.float32) / step + threshold[...,None] + 0.5), 0, levels - 1)
    return np.round(indices * step).astype(np.uint8)
