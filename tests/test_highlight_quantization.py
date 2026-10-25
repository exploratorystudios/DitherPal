import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from ditherpal import (DitherApp, DitherWorker, pixel_art_cel_shade, _light_cell_features,
                      _palette_color_cost, _palette_color_families)


class HighlightQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def source(self, shape=(64, 64)):
        image = np.full((*shape, 3), 25, np.float32)
        image[8:50, 14] = [80, 210, 245]
        image[24, 26:57] = [255, 205, 95]
        return image

    def assert_grid(self, image, block):
        for y in range(0, image.shape[0], block):
            for x in range(0, image.shape[1], block):
                cell = image[y:y+block, x:x+block]
                np.testing.assert_array_equal(cell, np.broadcast_to(cell[0, 0], cell.shape))

    def test_thin_lights_are_quantized_to_custom_palette_on_the_same_grid(self):
        source = self.source()
        palette = np.array([[20, 20, 20], [70, 190, 240], [240, 180, 60]], np.uint8)
        baseline = pixel_art_cel_shade(source, 4, shading_dither=0, palette=palette, preserve_highlights=False)
        result = pixel_art_cel_shade(source, 4, shading_dither=0, palette=palette)
        self.assert_grid(result, 4)
        self.assertGreater(int(result[20, 14, 2]), int(baseline[20, 14, 2]))
        self.assertGreater(int(result[24, 40, 0]), int(baseline[24, 40, 0]))
        for color in np.unique(result.reshape(-1, 3), axis=0):
            self.assertTrue((palette == color).all(axis=1).any())
        # These source colors are deliberately absent from the palette.
        self.assertFalse((result == [80, 210, 245]).all(axis=2).any())
        self.assertFalse((result == [255, 205, 95]).all(axis=2).any())

    def test_palette_budget_and_partial_edge_blocks_are_preserved(self):
        for bands in [4, 8, 24]:
            with self.subTest(bands=bands):
                result = pixel_art_cel_shade(self.source((63, 67)), 5, num_colors=bands)
                self.assertEqual(result.shape, (63, 67, 3))
                self.assertLessEqual(len(np.unique(result.reshape(-1, 3), axis=0)), bands)
                self.assert_grid(result, 5)

    def test_spark_is_not_deleted_as_an_orphan_or_copied_from_source(self):
        source = np.full((32, 32, 3), 20, np.float32)
        source[19, 19] = [245, 245, 245]
        palette = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
        result = pixel_art_cel_shade(source, 4, shading_dither=0, palette=palette)
        np.testing.assert_array_equal(result[16:20, 16:20], 255)
        self.assertEqual(int((result[..., 0] == 255).sum()), 16)
        self.assert_grid(result, 4)

    def test_uniform_shading_is_unchanged_and_grayscale_stays_quantized(self):
        source = np.full((32, 32, 3), [50, 100, 150], np.float32)
        a = pixel_art_cel_shade(source, 4, preserve_highlights=False)
        b = pixel_art_cel_shade(source, 4)
        np.testing.assert_array_equal(a, b)
        gray = np.full((32, 32, 1), 20, np.float32)
        gray[19, 19] = 245
        result = pixel_art_cel_shade(gray, 4, palette=np.array([[0], [255]], np.uint8))
        self.assertEqual(result.shape, gray.shape)
        self.assertTrue(np.isin(result, [0, 255]).all())
        self.assertEqual(int(result[19, 19, 0]), 255)

    def test_colored_edge_without_perceived_brightness_is_not_a_light(self):
        skin = np.full((32, 32, 3), [150, 120, 125], np.uint8)
        skin[10:22, 15] = [110, 150, 200]
        _, confidence = _light_cell_features(skin, 4)
        self.assertEqual(float(confidence.max()), 0)
        dark = np.full((32, 32, 3), 25, np.uint8)
        dark[10:22, 15] = [80, 210, 245]
        _, confidence = _light_cell_features(dark, 4)
        self.assertGreater(float(confidence.max()), .5)

    def test_warm_material_prefers_a_warm_shade_over_a_closer_blue(self):
        # Lab measurements from the reported finger artifact. Euclidean distance
        # chooses the blue because its lightness happens to be closer.
        skin = np.array([[138, 142, 126]], np.float32)
        colors = np.array([[131, 118, 109], [95, 142, 139]], np.float32)
        self.assertEqual(int(((skin[:, None]-colors)**2).sum(axis=2).argmin()), 0)
        self.assertEqual(int(_palette_color_cost(skin, colors).argmin()), 1)
        # An intentionally restricted custom palette must still have a valid
        # match even if none of its colors belong to the source hue family.
        self.assertTrue(np.isfinite(_palette_color_cost(skin, colors[:1])).all())

    def test_dither_ramps_do_not_mix_warm_skin_and_blue_but_allow_neutral_shades(self):
        colors = np.array([[138, 142, 126], [131, 118, 109], [140, 128, 128]], np.float32)
        families = _palette_color_families(colors)
        self.assertFalse(families[0, 1])
        self.assertTrue(families[0, 2])
        self.assertTrue(families[1, 2])
        self.assertTrue(_palette_color_families(np.array([[0], [255]], np.float32)).all())

    def test_still_pipeline_cannot_add_source_rgb_after_dithering(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.png', Path(directory)/'result.png'
            Image.fromarray(self.source().astype(np.uint8)).save(source)
            worker = DitherWorker(str(source), str(output), 'hybrid bayer + floyd-steinberg',
                                  4, 1, 1, color_mode='full_color', cel_shade=True, hybrid_cell=4)
            errors = []
            worker.error.connect(errors.append)
            with patch.object(worker, 'apply_dithering', return_value=np.full((64, 64, 3), [70, 40, 100], np.uint8)):
                worker.run()
            self.assertEqual(errors, [])
            with Image.open(output) as result:
                np.testing.assert_array_equal(np.array(result), np.broadcast_to([70, 40, 100], (64, 64, 3)))

    def test_dither_styles_receive_the_option_before_quantization(self):
        depth = np.full((64, 64), .5, np.float32)
        for style in ['off', 'retro', 'high_res']:
            worker = DitherWorker('', '', 'hybrid bayer + floyd-steinberg', 4, 1, 1,
                                  color_mode='full_color', cel_shade=True, hybrid_cell=4,
                                  pixel_art_style=style, preserve_highlights=False)
            with patch('ditherpal.pixel_art_cel_shade', wraps=pixel_art_cel_shade) as quantize:
                result = worker.apply_dithering(self.source(), worker.method, worker.levels, depth=depth)
            self.assertFalse(quantize.call_args.kwargs['preserve_highlights'])
            self.assert_grid(result, 4)

    def test_gui_setting_is_persisted_and_passed_to_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory)/'settings.ini'), QSettings.Format.IniFormat)
            with patch('ditherpal.QSettings', return_value=settings):
                window = DitherApp()
            try:
                window.pixel_art_combo.setCurrentIndex(1)
                self.assertTrue(window.light_details_check.isEnabled())
                window.light_details_check.setChecked(False)
                self.assertFalse(settings.value('preserve_highlights', type=bool))
                source = Path(directory)/'source.png'
                Image.fromarray(self.source().astype(np.uint8)).save(source)
                window.current_image_path = str(source)
                with patch('ditherpal.DitherWorker') as worker:
                    window.process_image()
                self.assertFalse(worker.call_args.kwargs['preserve_highlights'])
                window.worker_thread = None
            finally:
                window.worker_thread = None
                window.close()
                window.deleteLater()


if __name__ == '__main__':
    unittest.main()
