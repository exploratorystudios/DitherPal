"""Unproject depth into a triangle mesh and save portable 3D artifacts."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    width: int
    height: int
    focal: float


def build_mesh(z, fov=55.0, mesh_size=256, max_depth_jump=0.12):
    """Camera coordinates: X right, Y down, Z forward; vertices lie on source camera rays."""
    if z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Depth must be a positive finite HxW array, at least 2x2")
    if not 5 <= fov <= 140 or mesh_size < 2 or max_depth_jump <= 0:
        raise ValueError("Invalid field of view, mesh size, or depth-jump threshold")
    height, width = z.shape
    scale = min(1, mesh_size / max(width, height))
    grid_w, grid_h = max(2, round(width * scale)), max(2, round(height * scale))
    x, y = np.meshgrid(np.linspace(0, width - 1, grid_w, dtype=np.float32),
                       np.linspace(0, height - 1, grid_h, dtype=np.float32))
    depth = cv2.remap(z, x, y, cv2.INTER_LINEAR)
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    vertices = np.stack(((x - (width - 1) / 2) * depth / focal,
                         (y - (height - 1) / 2) * depth / focal, depth), axis=-1).reshape(-1, 3)
    uv = np.stack((x / (width - 1), y / (height - 1)), axis=-1).reshape(-1, 2)
    indices = np.arange(grid_w * grid_h).reshape(grid_h, grid_w)
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    # Front faces point toward the source camera (negative Z), including in OBJ exports.
    faces = np.concatenate((np.stack((a, c, b), axis=-1).reshape(-1, 3),
                            np.stack((b, c, d), axis=-1).reshape(-1, 3)))
    triangle_z = vertices[faces, 2]
    keep = triangle_z.max(axis=1) / triangle_z.min(axis=1) - 1 <= max_depth_jump
    return Mesh(np.ascontiguousarray(vertices, np.float32), np.ascontiguousarray(uv, np.float32),
                np.ascontiguousarray(faces[keep], np.int32), width, height, float(focal))


def write_ply(path, mesh, texture):
    """Binary PLY with RGB vertex colors and actual triangle faces (not a depth PNG)."""
    xs = np.clip(np.rint(mesh.uv[:, 0] * (texture.shape[1] - 1)).astype(int), 0, texture.shape[1] - 1)
    ys = np.clip(np.rint(mesh.uv[:, 1] * (texture.shape[0] - 1)).astype(int), 0, texture.shape[0] - 1)
    vertices = np.empty(len(mesh.vertices), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    vertices["xyz"], vertices["rgb"] = mesh.vertices, texture[ys, xs]
    faces = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    faces["count"], faces["indices"] = 3, mesh.faces
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment DitherPal neural zoom; camera X right Y down Z forward; arbitrary units\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with Path(path).open("wb") as output:
        output.write(header.encode("ascii"))
        output.write(vertices.tobytes())
        output.write(faces.tobytes())


def write_obj(folder, mesh):
    """Textured OBJ in conventional Y-up / camera-looking-down-minus-Z coordinates."""
    folder = Path(folder)
    (folder / "scene.mtl").write_text("newmtl photo\nKd 1 1 1\nKa 1 1 1\nmap_Kd texture.png\n")
    with (folder / "scene.obj").open("w") as output:
        output.write("mtllib scene.mtl\nusemtl photo\n")
        # Dense meshes have millions of lines; format them in bulk rather than one by one.
        for rows, line in ((mesh.vertices.astype(np.float64) * [1, -1, -1], "v %.7g %.7g %.7g\n"),
                           (np.column_stack((mesh.uv[:, 0], 1 - mesh.uv[:, 1].astype(np.float64))), "vt %.7g %.7g\n"),
                           (np.repeat(mesh.faces + 1, 2, axis=1).tolist(), "f %d/%d %d/%d %d/%d\n")):
            rows = rows if isinstance(rows, list) else rows.tolist()
            for start in range(0, len(rows), 65536):
                chunk = rows[start:start + 65536]
                output.write((line * len(chunk)) % tuple(value for row in chunk for value in row))
