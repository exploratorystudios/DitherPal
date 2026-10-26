import json
from pathlib import Path
import tempfile
import os
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ditherpal_depth import layers
from ditherpal_depth.renderer import rasterize
from ditherpal_depth.scene import NeuralScene


def figure():
    """A body on the far background with an arm standing on a step in front of it."""
    near = np.zeros((64, 64), np.float32)
    near[12:56, 16:48] = .5
    near[28:36, 20:44] = .9
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[:] = [20, 60, 220]
    rgb[near == .5] = [200, 40, 40]
    rgb[near == .9] = [240, 220, 60]
    return rgb, near


def paint_green(rgb, mask, *args, **kwargs):
    return np.where(mask[..., None], np.array([30, 200, 60], np.uint8), rgb)


def _die(job):
    os._exit(1)   # A worker killed mid-frame, as under memory pressure.


def _hang(job):
    time.sleep(3600)   # A worker stuck on an inherited lock.


class HiddenLayerTests(unittest.TestCase):
    def test_masks_cover_the_subject_and_the_band_under_its_front_pieces(self):
        _, near = figure()
        masks = layers.layer_masks(near)
        self.assertEqual(set(masks), {'sky', 'inner'})
        self.assertTrue(masks['sky'][near > 0].all())
        inner = masks['inner']
        self.assertTrue(inner.any())
        self.assertFalse(inner[near == 0].any())
        # Only the arm and its edge band, not the rest of the body.
        self.assertTrue(inner[28:36, 20:44].any())
        self.assertFalse(inner[14:20, 18:46].any())

    def test_flat_depth_has_no_layers(self):
        self.assertEqual(layers.layer_masks(np.full((32, 32), .5, np.float32)), {})

    def test_hidden_depth_is_never_pulled_forward(self):
        _, near = figure()
        mask = layers.layer_masks(near)['inner']
        hidden = layers.layer_near(near, mask)
        self.assertTrue((hidden <= near+1e-6).all())
        self.assertTrue((hidden[mask] < .9).all())
        np.testing.assert_array_equal(hidden[~mask], near[~mask])

    def test_sheets_do_not_tear(self):
        _, near = figure()
        mask = layers.layer_masks(near)['sky']
        z = (1/(.25+.75*near)).astype(np.float32)
        mesh = layers.sheet(z, mask)
        self.assertGreater(len(mesh.faces), 0)
        self.assertGreater(np.ptp(mesh.vertices[mesh.faces, 2], axis=1).max(), .1)

    def test_gaps_show_hidden_layers_and_visible_pixels_are_kept(self):
        rgb, near = figure()
        scene = NeuralScene(Image.fromarray(rgb), near)
        scene.prepare_layers(paint_green)
        self.assertEqual([layer.name for layer in scene.layers], ['inner', 'sky'])
        mesh = scene.mesh
        camera = np.array([.02, 0, .14], np.float32)
        raw, z = rasterize(mesh.vertices, mesh.uv, mesh.faces, scene.texture,
                           mesh.width, mesh.height, mesh.focal, camera, 0.)
        missing = ~np.isfinite(z)
        self.assertTrue(missing.any())
        frame, depth, shown = scene._composite(raw, z, raw.copy(), z.copy(), scene.layers,
                                               mesh.width, mesh.height, mesh.focal, camera, 1)
        np.testing.assert_array_equal(frame[~missing], raw[~missing])
        self.assertFalse(shown[~missing].any())
        self.assertGreater((shown & missing).sum(), missing.sum()//2)
        self.assertTrue((frame[shown] == [30, 200, 60]).all(axis=1).any())
        self.assertTrue(np.isfinite(depth[shown]).all())

    def test_layers_are_reported_and_written(self):
        rgb, near = figure()
        scene = NeuralScene(Image.fromarray(rgb), near)
        scene.prepare_layers(paint_green)
        with tempfile.TemporaryDirectory() as directory:
            scene.write(directory, 'figure', .14)
            frames = list(scene.frames(directory, count=4))
            self.assertEqual(len(frames), 4)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(set(report['hidden_layers']['layers']), {'inner', 'sky'})
            self.assertGreater(max(report['hidden_layers']['pixels_per_frame']), 0)
            for name in ('hidden_sky_texture.png', 'hidden_inner_mask.png', 'hidden_layer_mask.png'):
                self.assertTrue((Path(directory)/name).is_file(), name)
            raw_only = list(scene.frames(directory, count=2, fill_holes=False))
            self.assertIsNone(json.loads((Path(directory)/'report.json').read_text())['hidden_layers'])
            self.assertEqual(len(raw_only), 2)

    def test_parallel_camera_angles_match_serial_frames(self):
        rgb, near = figure()
        scene = NeuralScene(Image.fromarray(rgb), near)
        scene.prepare_layers(paint_green)
        with tempfile.TemporaryDirectory() as directory:
            serial = list(scene.frames(directory, count=4))
            parallel = list(scene.frames(directory, count=4, workers=2))
            self.assertEqual(json.loads((Path(directory)/'report.json').read_text())['render_workers'], 2)
            self.assertTrue((Path(directory)/'hidden_layer_mask.png').is_file())
        self.assertEqual(len(parallel), 4)
        for (a, a_near), (b, b_near) in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a_near, b_near)

    def test_a_dead_worker_falls_back_to_serial_rendering(self):
        rgb, near = figure()
        scene = NeuralScene(Image.fromarray(rgb), near)
        with tempfile.TemporaryDirectory() as directory:
            serial = list(scene.frames(directory, count=4, fill_holes=False))
            with patch('ditherpal_depth.scene._render_job', _die):
                recovered = list(scene.frames(directory, count=4, fill_holes=False, workers=2))
        self.assertEqual(len(recovered), 4)
        for (a, _), (b, _) in zip(serial, recovered):
            np.testing.assert_array_equal(a, b)

    def test_a_stuck_worker_times_out_and_falls_back_to_serial_rendering(self):
        rgb, near = figure()
        scene = NeuralScene(Image.fromarray(rgb), near)
        with tempfile.TemporaryDirectory() as directory:
            serial = list(scene.frames(directory, count=3, fill_holes=False))
            with patch('ditherpal_depth.scene._render_job', _hang), patch('ditherpal_depth.scene.FRAME_TIMEOUT', 2):
                recovered = list(scene.frames(directory, count=3, fill_holes=False, workers=2))
        for (a, _), (b, _) in zip(serial, recovered):
            np.testing.assert_array_equal(a, b)

    def test_smooth_seams_split_inner_gaps_and_leave_sky_gaps(self):
        depth = np.full((40, 60), 2., np.float32)       # Rear subject layer.
        depth[:, :20] = 1.                               # Front piece.
        depth[:, 45:] = 4.                               # Far background.
        depth[:, 20:30] = np.inf                         # Inner gap.
        depth[:, 40:45] = np.inf                         # Gap onto the far background.
        raw = np.zeros((40, 60, 3), np.uint8)
        raw[:, :20] = [240, 220, 60]
        raw[:, 30:40] = [200, 40, 40]
        raw[:, 45:] = [20, 60, 220]
        repaired = raw.copy()
        repaired[:, 40:45] = [20, 60, 220]
        result = layers.smooth_seams(raw, depth, repaired, 1, 3., 2.)
        self.assertTrue((result[:, 21] == [240, 220, 60]).all())
        self.assertTrue((result[:, 28] == [200, 40, 40]).all())
        np.testing.assert_array_equal(result[:, 40:], repaired[:, 40:])


if __name__ == '__main__':
    unittest.main()
