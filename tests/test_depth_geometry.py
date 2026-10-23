import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ditherpal_depth.artifacts import write_viewer
from ditherpal_depth.model import depth_to_z, download_model, preprocess
from ditherpal_depth.geometry import Mesh, build_mesh, write_obj, write_ply
from ditherpal_depth.renderer import ordered_dither, rasterize, render


class GeometryTests(unittest.TestCase):
    def test_inverse_depth_has_correct_direction_and_constant_maps_are_safe(self):
        prediction = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        z, _ = depth_to_z(prediction, 1, 4)
        self.assertAlmostEqual(float(z[0, 0]), 4)
        self.assertAlmostEqual(float(z[-1, -1]), 1)
        flat, _ = depth_to_z(np.ones((4, 4), np.float32))
        self.assertTrue(np.isfinite(flat).all())
        self.assertTrue((flat > 0).all())
        with self.assertRaises(ValueError):
            depth_to_z(np.full((4, 4), np.nan))

    def test_unprojection_reprojects_to_source_pixels(self):
        mesh = build_mesh(np.full((24, 32), 2, np.float32), mesh_size=64)
        x = mesh.focal * mesh.vertices[:, 0] / mesh.vertices[:, 2] + 15.5
        y = mesh.focal * mesh.vertices[:, 1] / mesh.vertices[:, 2] + 11.5
        np.testing.assert_allclose(x, mesh.uv[:, 0] * 31, atol=1e-5)
        np.testing.assert_allclose(y, mesh.uv[:, 1] * 23, atol=1e-5)

    def test_flat_mesh_reconstructs_texture_without_a_2d_warp(self):
        texture = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        mesh = build_mesh(np.full((24, 32), 2, np.float32), mesh_size=64)
        image, missing, _ = render(mesh, texture)
        self.assertFalse(missing.any())
        np.testing.assert_allclose(image, texture, atol=1)

    def test_depth_boundaries_do_not_become_stretched_triangles(self):
        z = np.ones((8, 8), np.float32)
        z[:, 4:] = 3
        mesh = build_mesh(z, mesh_size=8, max_depth_jump=.12)
        face_depth = mesh.vertices[mesh.faces, 2]
        self.assertTrue(np.all(face_depth.max(axis=1) == face_depth.min(axis=1)))
        self.assertGreater(len(mesh.faces), 0)
        self.assertLess(len(mesh.faces), 2*7*7)

    def test_exported_surface_front_faces_point_toward_source_camera(self):
        mesh = build_mesh(np.ones((8, 8), np.float32))
        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        self.assertTrue((normals[:, 2] < 0).all())

    def test_arbitrary_scene_units_do_not_change_neutral_render(self):
        texture = np.full((8, 8, 3), 150, np.uint8)
        for depth in (0.001, 1, 1000):
            mesh = build_mesh(np.full((8, 8), depth, np.float32))
            image, missing, _ = render(mesh, texture)
            self.assertFalse(missing.any())
            np.testing.assert_allclose(image, texture, atol=1)

    def test_camera_dolly_magnifies_near_geometry_more_than_far(self):
        magnifications = []
        for depth in (1, 4):
            mesh = build_mesh(np.full((8, 8), depth, np.float32))
            x, _, z = mesh.vertices[-1]
            original = mesh.focal * x / z
            moved = mesh.focal * x / (z-.2)
            magnifications.append(moved/original)
        self.assertAlmostEqual(float(magnifications[0]), 1.25, places=5)
        self.assertAlmostEqual(float(magnifications[1]), 4/3.8, places=5)
        self.assertGreater(magnifications[0], magnifications[1])

    def test_z_buffer_selects_nearest_face_independent_of_submission_order(self):
        pixels = np.array([[3, 3], [17, 3], [10, 17]], np.float32)
        vertices = []
        for depth in (1, 3):
            vertices.extend(np.column_stack(((pixels-10)*depth/10, np.full(3, depth))))
        vertices = np.array(vertices, np.float32)
        uv = np.array([[0, 0]]*3 + [[1, 0]]*3, np.float32)
        texture = np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8)
        faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
        for ordered in (faces, faces[::-1].copy()):
            image, depth = rasterize(vertices, uv, ordered, texture, 21, 21, 10., np.zeros(3, np.float32), 0.)
            np.testing.assert_array_equal(image[8, 10], [255, 0, 0])
            self.assertAlmostEqual(float(depth[8, 10]), 1)

    def test_texture_interpolation_is_perspective_correct(self):
        pixels = np.array([[2, 2], [18, 2], [2, 18]], np.float32)
        z = np.array([1, 2, 4], np.float32)
        vertices = np.column_stack(((pixels-10)*z[:, None]/10, z)).astype(np.float32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
        y, x = np.indices((32, 32))
        texture = np.stack((x/31*255, y/31*255, x*0), axis=-1).round().astype(np.uint8)
        image, _ = rasterize(vertices, uv, np.array([[0, 1, 2]], np.int32), texture,
                             21, 21, 10., np.zeros(3, np.float32), 0.)
        inverse_z = .5/1 + .25/2 + .25/4
        np.testing.assert_allclose(image[6, 6, :2], [255*(.25/2)/inverse_z, 255*(.25/4)/inverse_z], atol=2)

    def test_optional_fill_keeps_raw_coverage_diagnostics(self):
        z = np.ones((16, 16), np.float32)
        z[:, 8:] = 3
        mesh = build_mesh(z, mesh_size=16)
        texture = np.full((16, 16, 3), 180, np.uint8)
        _, missing, raw = render(mesh, texture, camera=(.08, 0, 0), fill_holes=True)
        self.assertTrue(missing.any())
        self.assertTrue((raw[missing] == 0).all())

    def test_exports_contain_geometry_and_self_contained_viewer(self):
        mesh = build_mesh(np.ones((4, 4), np.float32), mesh_size=4)
        texture = np.full((4, 4, 3), 100, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_ply(folder/'scene.ply', mesh, texture)
            write_obj(folder, mesh)
            write_viewer(folder/'viewer.html', mesh, texture, '</script>example', .14)
            ply = (folder/'scene.ply').read_bytes()
            header, binary = ply.split(b'end_header\n', 1)
            self.assertIn(b'element vertex 16', header)
            self.assertEqual(len(binary), len(mesh.vertices)*15 + len(mesh.faces)*13)
            self.assertIn('map_Kd texture.png', (folder/'scene.mtl').read_text())
            self.assertEqual(sum(line.startswith('f ') for line in (folder/'scene.obj').read_text().splitlines()), len(mesh.faces))
            html = (folder/'viewer.html').read_text()
            self.assertIn('data:image/png;base64,', html)
            self.assertNotIn('__SCENE_DATA__', html)
            self.assertNotIn('</script>example', html)

    def test_download_is_verified_and_failed_download_is_removed(self):
        payload = b'test weights'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.onnx'
            with patch('urllib.request.urlopen', return_value=io.BytesIO(payload)):
                with self.assertRaises(ValueError):
                    download_model(path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.download').exists())
            with patch('urllib.request.urlopen', return_value=io.BytesIO(payload)):
                with patch('ditherpal_depth.model.MODEL_SHA256', hashlib.sha256(payload).hexdigest()):
                    download_model(path)
            self.assertEqual(path.read_bytes(), payload)

    def test_preprocessing_uses_rgb_normalization_and_patch_multiple(self):
        rgb = np.full((120, 240, 3), 255, np.uint8)
        tensor = preprocess(rgb)
        self.assertEqual(tensor.shape[0:2], (1, 3))
        self.assertEqual(tensor.shape[-2] % 14, 0)
        self.assertEqual(tensor.shape[-1] % 14, 0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), (1-.485)/.229, places=5)
        self.assertEqual(preprocess(rgb, fixed_shape=(518, 518)).shape, (1, 3, 518, 518))

    def test_post_dither_has_a_fixed_palette(self):
        rgb = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        image = ordered_dither(rgb, 4)
        self.assertTrue(np.isin(image, [0, 85, 170, 255]).all())


if __name__ == '__main__':
    unittest.main()
