import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ditherpal import (DitherWorker, pixel_art_cel_shade, high_res_pixel_dither,
                      retro_pixel_dither)
from ditherpal_depth.geometry import build_mesh
from ditherpal_depth.renderer import rasterize, rasterize_alpha
from ditherpal_depth.repairs import DonorRepair, _strips
from ditherpal_depth.seam_art import SeamPixelArt


class BackgroundOnlySeamsTests(unittest.TestCase):
    def source(self, scale=1):
        depth = np.ones((48, 64), np.float32)
        depth[:, 32:] = 3
        texture = np.full((48*scale, 64*scale, 3), [240, 30, 20], np.uint8)
        y, x = np.indices((48*scale, 32*scale))
        texture[:, 32*scale:] = np.where(((x//scale+y//scale) % 2)[..., None],
                                         [20, 60, 220], [40, 80, 200])
        return build_mesh(depth, mesh_size=64), texture, depth

    def art(self, style, pattern=None):
        return SeamPixelArt(style, (64, 48), 4, 4, (12, 16), 24, .85, True,
                            pixel_art_cel_shade,
                            pattern or (high_res_pixel_dither if style == 'high_res' else retro_pixel_dither))

    def test_no_foreground_repair_geometry_is_constructed(self):
        mesh, texture, depth = self.source()
        arrays = [v.copy() for v in (mesh.vertices, mesh.uv, mesh.faces)]
        original = texture.copy()
        with patch('ditherpal_depth.repairs._strips', wraps=_strips) as strips:
            repair = DonorRepair(mesh, texture, depth)
        self.assertEqual([call.args[2] for call in strips.call_args_list], ['background'])
        self.assertEqual(len(repair.foreground.faces), 0)
        for a, b in zip(arrays, (mesh.vertices, mesh.uv, mesh.faces)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(texture, original)

    def test_near_fragment_rejection_reveals_farther_background_instead_of_occluding_it(self):
        vertices, uv, faces = [], [], []
        for z, u in ((1., 0.), (3., 1.)):
            offset = len(vertices)
            xy = np.array([[1, 1], [8, 1], [1, 8], [8, 8]], np.float32)
            vertices.extend(np.column_stack(((xy-4.5)*z/10, np.full(4, z))))
            uv.extend([[u, 0.]]*4)
            faces.extend([(offset, offset+2, offset+1), (offset+1, offset+2, offset+3)])
        texture = np.array([[[240, 20, 20], [20, 60, 220]], [[240, 20, 20], [20, 60, 220]]], np.uint8)
        limit = np.zeros((10, 10), np.float32)
        limit[3:7, 3:7] = 2
        args = (np.asarray(vertices, np.float32), np.asarray(uv, np.float32),
                np.asarray(faces, np.int32), texture, 10, 10, 10., np.zeros(3, np.float32), 0.,
                np.ones(8, np.float32))
        result, depth, _ = rasterize_alpha(*args, limit)
        np.testing.assert_array_equal(result[3:7, 3:7], np.broadcast_to([20, 60, 220], (4, 4, 3)))
        np.testing.assert_array_equal(depth[3:7, 3:7], 3)
        np.testing.assert_array_equal(result[2, 2], [240, 20, 20])
        np.testing.assert_array_equal(depth[2, 2], 1)

    def test_foreground_colors_cannot_change_background_art_in_either_mode_or_texture_scale(self):
        for style in ('high_res', 'retro'):
            for scale in (1, 2):
                with self.subTest(style=style, scale=scale):
                    mesh, texture, depth = self.source(scale)
                    camera = np.array([0, 0, .14], np.float32)
                    images, raw_depths = [], []
                    for color in ([240, 30, 20], [255, 255, 0]):
                        source = texture.copy()
                        source[:, :32*scale] = color
                        repair = DonorRepair(mesh, source, depth, pixel_art=self.art(style))
                        for item in repair.metadata+repair.extended_metadata:
                            context = item['art_context']
                            if context is not None:
                                self.assertTrue(all(tuple(c) in {(20, 60, 220), (40, 80, 200)}
                                                    for c in context['palette']))
                        raw, z = rasterize(mesh.vertices, mesh.uv, mesh.faces, source,
                                            mesh.width, mesh.height, mesh.focal, camera, 0.)
                        image, _, _, feather = repair.render(raw, z, mesh.focal, camera)
                        np.testing.assert_array_equal(image[np.isfinite(z)], raw[np.isfinite(z)])
                        self.assertFalse(feather.any())
                        images.append(image)
                        raw_depths.append(z)
                    missing = ~np.isfinite(raw_depths[0])
                    np.testing.assert_array_equal(images[0][missing], images[1][missing])

    def test_selected_mode_and_exact_palette_are_used_and_pattern_cache_is_bounded(self):
        mesh, texture, depth = self.source()
        for style, routine in [('high_res', high_res_pixel_dither), ('retro', retro_pixel_dither)]:
            with self.subTest(style=style):
                with patch(__name__+'.'+routine.__name__, wraps=routine) as effect:
                    art = self.art(style, effect)
                    repair = DonorRepair(mesh, texture, depth, pixel_art=art)
                    for i in range(12):
                        atlas = repair._fixed.copy()
                        # Distinct healed inputs exercise eviction, without
                        # changing the allowed source-art palette.
                        atlas = np.minimum(atlas.astype(np.uint16)+i, 255).astype(np.uint8)
                        result = art.dither(atlas, repair.metadata)
                    self.assertGreater(effect.call_count, 0)
                    before = effect.call_count
                    np.testing.assert_array_equal(result, art.dither(atlas, repair.metadata))
                    self.assertEqual(effect.call_count, before)
                for item in repair.metadata:
                    context = item['art_context']
                    if context is None:
                        continue
                    self.assertLessEqual(len(context['cache']), 8)
                    allowed = {tuple(c) for c in context['palette']}
                    for grid in context['cache'].values():
                        self.assertTrue(all(tuple(c) in allowed for c in grid.reshape(-1, 3)))

    def test_cancellation_during_a_frame_leaves_original_texture_and_mesh_untouched(self):
        mesh, texture, depth = self.source()
        interrupted = False
        repair = DonorRepair(mesh, texture, depth, cancelled=lambda: interrupted)
        original = texture.copy()
        camera = np.array([0, 0, .14], np.float32)
        raw, z = rasterize(mesh.vertices, mesh.uv, mesh.faces, texture,
                            mesh.width, mesh.height, mesh.focal, camera, 0.)
        interrupted = True
        with self.assertRaises(InterruptedError):
            repair.render(raw, z, mesh.focal, camera)
        np.testing.assert_array_equal(texture, original)

    def test_both_pixel_modes_reach_the_seam_pipeline_for_all_animation_exports(self):
        mesh, texture, depth = self.source()
        near = np.clip((1/depth-.25)/.75, 0, 1).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.png'
            Image.fromarray(texture).save(source)
            for style in ('high_res', 'retro'):
                for format in ('gif', 'mp4', 'svg'):
                    with self.subTest(style=style, format=format):
                        output = root/f'{style}.{format}'
                        worker = DitherWorker(str(source), str(output), 'hybrid bayer + floyd-steinberg',
                            2, 1, 1, color_mode='full_color', cel_shade=True, hybrid_cell=4,
                            pixel_art_style=style, zoom_format=format)
                        errors = []
                        worker.error.connect(errors.append)
                        with patch('ditherpal.neural_depth_for_file', return_value=near):
                            with patch('ditherpal.ZOOM_FRAMES', 2):
                                worker.run()
                        self.assertEqual(errors, [])
                        self.assertTrue(output.is_file())
                        folder = Path(worker.mesh_viewer_path).parent
                        report = json.loads((folder/'report.json').read_text())
                        self.assertEqual(report['gap_repair']['pixel_art'], style)
                        self.assertEqual(report['gap_repair']['foreground_seam_triangles'], 0)
                        with np.load(folder/'repair_layers.npz') as layers:
                            self.assertEqual(len(layers['foreground_faces']), 0)


if __name__ == '__main__':
    unittest.main()
