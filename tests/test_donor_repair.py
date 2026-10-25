import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ditherpal_depth.geometry import build_mesh
from ditherpal_depth.renderer import rasterize, rasterize_alpha
from ditherpal_depth.repairs import DonorRepair, _boundaries, _donors, _heal_patch
from ditherpal_depth.scene import NeuralScene


class DonorRepairTests(unittest.TestCase):
    def source(self, scale=1):
        depth = np.ones((48, 64), np.float32)
        depth[:, 32:] = 3
        texture = np.empty((48*scale, 64*scale, 3), np.uint8)
        texture[:, :32*scale] = [240, 30, 20]
        texture[:, 32*scale:] = [20, 60, 220]
        return build_mesh(depth, mesh_size=64), texture, depth

    def test_donors_use_background_only_at_native_and_larger_texture_sizes(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                mesh, texture, depth = self.source(scale)
                repair = DonorRepair(mesh, texture, depth)
                self.assertGreater(len(repair.locations), 0)
                atlas = repair.donors.texture
                colors = atlas[(atlas != 0).any(axis=2)]
                # OpenCV's gradient solver can round a constant channel down by
                # one. It must still contain only the blue background, never red foreground.
                np.testing.assert_allclose(colors, np.broadcast_to([20, 60, 220], colors.shape), rtol=0, atol=1)
                self.assertGreater(len(repair.foreground.faces), 0)
                for layer in [repair.background, repair.foreground, repair.donors]:
                    z = layer.vertices[layer.faces, 2]
                    self.assertTrue((z.max(axis=1)/z.min(axis=1) < 1.13).all())
                # Sampling density follows the baked texture rather than losing
                # its detail when AI geometry is rendered at a smaller size.
                self.assertGreaterEqual(atlas.shape[0], 16*scale)

    def test_alpha_interpolation_is_perspective_correct_and_preserves_rgb_and_depth(self):
        pixels = np.array([[2, 2], [18, 2], [2, 18]], np.float32)
        z = np.array([1, 2, 4], np.float32)
        vertices = np.column_stack(((pixels-10)*z[:, None]/10, z)).astype(np.float32)
        uv = np.zeros((3, 2), np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        texture = np.full((2, 2, 3), 255, np.uint8)
        args = (vertices, uv, faces, texture, 21, 21, 10., np.zeros(3, np.float32), 0.)
        rgb, depth = rasterize(*args)
        color, layered_depth, opacity = rasterize_alpha(*args, np.array([1, 0, 0], np.float32))
        np.testing.assert_array_equal(color, rgb)
        np.testing.assert_array_equal(layered_depth, depth)
        self.assertAlmostEqual(float(opacity[6, 6]), (.5/1)/(.5/1+.25/2+.25/4), places=6)
        self.assertTrue((opacity[~np.isfinite(depth)] == 0).all())

    def test_donors_keep_two_dimensional_background_texture_detail(self):
        mesh, texture, depth = self.source(scale=2)
        y, x = np.indices(texture.shape[:2])
        texture[:, 64:] = np.where(((x[:, 64:]//2+y[:, 64:]//2) % 2)[..., None],
                                   [20, 60, 220], [30, 80, 200])
        patches, _ = _donors(mesh, texture, depth, _boundaries(mesh), lambda: False, None)
        self.assertTrue(any(np.ptp(p['healed'], axis=0).max() > 0
                            and np.ptp(p['healed'], axis=1).max() > 0 for p in patches))

    def test_healing_matches_background_tone_without_sampling_foreground(self):
        copied = np.full((16, 25, 3), [20, 60, 180], np.uint8)
        across = np.linspace(-3, 12, 16, dtype=np.float32)
        safe = np.broadcast_to((across < -.5)[:, None], copied.shape[:2])
        original = np.full_like(copied, [240, 30, 20])
        original[safe] = [50, 80, 200]
        healed = _heal_patch(copied, original, safe, across)
        np.testing.assert_array_equal(healed[0], np.broadcast_to([50, 80, 200], healed[0].shape))
        # The unknown side may contain a foreground object: its colors must
        # never influence the background tone correction or gradient solve.
        original[~safe] = [0, 255, 0]
        np.testing.assert_array_equal(healed, _heal_patch(copied, original, safe, across))
        self.assertLess(int(healed[..., 0].max()), 100)
        self.assertGreater(int(healed[..., 2].min()), 170)
        original[safe] = 255
        bounded = _heal_patch(copied, original, safe, across)
        np.testing.assert_array_equal(bounded[0], np.broadcast_to([68, 108, 228], bounded[0].shape))

    def test_healing_is_prepared_once_and_not_recomputed_during_rendering(self):
        mesh, texture, depth = self.source()
        with patch('ditherpal_depth.repairs._heal_patch', wraps=_heal_patch) as heal:
            repair = DonorRepair(mesh, texture, depth)
            prepared = heal.call_count
            self.assertEqual(prepared, len(repair.locations))
            self.assertGreater(prepared, 0)
            for camera in (np.zeros(3, np.float32), np.array([0, 0, .14], np.float32)):
                raw, z = rasterize(mesh.vertices, mesh.uv, mesh.faces, texture,
                                   mesh.width, mesh.height, mesh.focal, camera, 0.)
                repair.render(raw, z, mesh.focal, camera)
            self.assertEqual(heal.call_count, prepared)

    def test_cancellation_during_healing_discards_incomplete_preparation(self):
        mesh, texture, depth = self.source()
        interrupted = False

        def heal(*args):
            nonlocal interrupted
            result = _heal_patch(*args)
            interrupted = True
            return result

        with patch('ditherpal_depth.repairs._heal_patch', side_effect=heal) as solve:
            with self.assertRaises(InterruptedError):
                DonorRepair(mesh, texture, depth, cancelled=lambda: interrupted)
        solve.assert_called_once()

    def test_patches_preserve_nearer_foreground_and_fill_missing_background(self):
        shape = (8, 8)
        raw = np.full((*shape, 3), [240, 30, 20], np.uint8)
        depth = np.ones(shape, np.float32)
        depth[:, 4:] = np.inf
        raw[:, 4:] = 0
        blank = (np.zeros_like(raw), np.full(shape, np.inf, np.float32), np.zeros(shape, np.float32))
        donor = (np.full_like(raw, [20, 60, 220]), np.full(shape, 3, np.float32), np.ones(shape, np.float32))
        repair = DonorRepair.__new__(DonorRepair)
        repair.background = SimpleNamespace(render=lambda *args: blank)
        repair.foreground = SimpleNamespace(render=lambda *args: blank)
        repair.donors = SimpleNamespace(render=lambda *args: donor)
        result, repaired_depth, opacity, _ = repair.render(raw, depth, 10., np.zeros(3, np.float32))
        np.testing.assert_array_equal(result[:, :4], raw[:, :4])
        np.testing.assert_array_equal(result[:, 4:], np.broadcast_to([20, 60, 220], result[:, 4:].shape))
        np.testing.assert_array_equal(repaired_depth[:, :4], 1)
        np.testing.assert_array_equal(repaired_depth[:, 4:], 3)
        np.testing.assert_array_equal(opacity[:, :4], 0)

    def test_selection_runs_once_and_frames_are_deterministic(self):
        mesh, texture, depth = self.source()
        near = np.clip((1/depth-.25)/.75, 0, 1).astype(np.float32)
        scene = NeuralScene(Image.fromarray(texture), near)
        with tempfile.TemporaryDirectory() as directory:
            with patch('ditherpal_depth.repairs._donors', wraps=_donors) as select:
                first = list(scene.frames(directory, count=4))
                atlas = scene.repairs.donors.texture
                second = list(scene.frames(directory, count=4))
                self.assertIs(scene.repairs.donors.texture, atlas)
            select.assert_called_once()
            for (a, _), (b, _) in zip(first, second):
                np.testing.assert_array_equal(a, b)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['gap_repair']['method'], 'background_healed_donor_patches')
            self.assertEqual(len(report['donor_pixels_per_frame']), 4)
            self.assertTrue((Path(directory)/'donor_patch_mask.png').is_file())

    def test_disabled_repair_renders_raw_mesh_and_does_not_select_patches(self):
        _, texture, depth = self.source()
        near = np.clip((1/depth-.25)/.75, 0, 1).astype(np.float32)
        scene = NeuralScene(Image.fromarray(texture), near)
        with tempfile.TemporaryDirectory() as directory:
            with patch('ditherpal_depth.scene.DonorRepair') as repair:
                frames = list(scene.frames(directory, count=2, fill_holes=False))
            repair.assert_not_called()
            mesh = scene.mesh
            expected, _ = rasterize(mesh.vertices, mesh.uv, mesh.faces, texture, mesh.width, mesh.height,
                                    mesh.focal, np.zeros(3, np.float32), 0.)
            np.testing.assert_array_equal(frames[0][0], expected)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertIsNone(report['gap_repair'])
            self.assertEqual(report['donor_pixels_per_frame'], [])

    def test_cancellation_before_and_during_patch_selection(self):
        mesh, texture, depth = self.source()
        with self.assertRaises(InterruptedError):
            DonorRepair(mesh, texture, depth, cancelled=lambda: True)
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 5

        with self.assertRaises(InterruptedError):
            _donors(mesh, texture, depth, _boundaries(mesh), cancelled, None)
        self.assertEqual(calls, 5)

    def test_flat_surfaces_have_no_artificial_seams_and_repair_assets_are_portable(self):
        mesh = build_mesh(np.ones((8, 8), np.float32))
        empty = DonorRepair(mesh, np.full((8, 8, 3), 100, np.uint8), np.ones((8, 8), np.float32))
        self.assertEqual(empty.locations, [])
        self.assertEqual(len(empty.background.faces), 0)
        self.assertEqual(len(empty.foreground.faces), 0)
        mesh, texture, depth = self.source()
        repair = DonorRepair(mesh, texture, depth)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            repair.write(folder)
            with np.load(folder/'repair_layers.npz') as data:
                np.testing.assert_array_equal(data['donors_alpha'], repair.donors.alpha)
                np.testing.assert_array_equal(data['foreground_faces'], repair.foreground.faces)
            with Image.open(folder/'donor_texture.png') as atlas:
                np.testing.assert_array_equal(np.array(atlas), repair.donors.texture)
            self.assertEqual(len(json.loads((folder/'donor_locations.json').read_text())), len(repair.locations))


if __name__ == '__main__':
    unittest.main()
