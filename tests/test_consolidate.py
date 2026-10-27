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

from ditherpal import DitherApp, DitherWorker


class ConsolidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        rgb = np.random.default_rng(7).integers(0, 256, (60, 90, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(self.source)

    def render(self, source, name, downscale=1, upsample=1, **kwargs):
        settings = dict(color_mode='full_color', cel_shade=True, hybrid_cell=6, restore_size=False)
        settings.update(kwargs)
        output = self.root/name
        worker = DitherWorker(str(source), str(output), 'hybrid bayer + floyd-steinberg', 2,
                              upsample, downscale, **settings)
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        with Image.open(output) as result:
            return np.array(result)

    def test_pixel_art_dithers_the_consolidated_image_at_1x(self):
        # Same as consolidating by hand (BOX down 3x, Lanczos up 2x) and dithering that file
        with Image.open(self.source) as image:
            small = image.resize((30, 20), Image.Resampling.BOX)
            small.resize((60, 40), Image.Resampling.LANCZOS).save(self.root/'prepared.png')
        expected = self.render(self.root/'prepared.png', 'expected.png')
        result = self.render(self.source, 'result.png', 3, 2, consolidate=True)
        self.assertEqual(result.shape, (40, 60, 3))
        np.testing.assert_array_equal(result, expected)
        # Without the option the same factors keep their usual meaning
        self.assertFalse(np.array_equal(self.render(self.source, 'plain.png', 3, 2), expected))

    def test_consolidated_result_can_return_to_the_original_size(self):
        for option in ('restore_size', 'restore_upscale'):
            with self.subTest(option=option):
                art = self.render(self.source, f'art_{option}.png', 3, 2, consolidate=True, **{option: True})
                self.assertEqual(art.shape, (60, 90, 3))
                # Art pixels stay whole blocks (6 consolidated px = 9 original px)
                cell = art[9:18, 9:18]
                np.testing.assert_array_equal(cell, np.broadcast_to(cell[0, 0], cell.shape))
                dither = self.render(self.source, f'fs_{option}.png', 3, 2, consolidate=True,
                                     cel_shade=False, **{option: True})
                self.assertEqual(dither.shape[:2], (60, 90))

    def test_gui_fits_pixel_art_to_the_consolidated_image_and_passes_the_option(self):
        settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        with patch('ditherpal.QSettings', return_value=settings):
            window = DitherApp()
        try:
            window.current_image_path = str(self.source)
            window.pixel_art_combo.setCurrentIndex(1)
            plain_block = window.hybrid_cell_spin.value()
            window.consolidate_check.setChecked(True)
            window.downscale_spin.setValue(3)
            window.upsample_spin.setValue(2)
            self.assertTrue(settings.value('consolidate', type=bool))
            # The factors are kept, and the block is sized for the 60x40 consolidated image
            self.assertEqual((window.downscale_spin.value(), window.upsample_spin.value()), (3, 2))
            self.assertLessEqual(window.hybrid_cell_spin.value(), plain_block)
            self.assertIn('Consolidated', window.image_label.text())
            with patch('ditherpal.DitherWorker') as worker:
                window.process_image()
            self.assertTrue(worker.call_args.kwargs['consolidate'])
            window.worker_thread = None
            # Factors stay editable while the option is on, even when only one is above 1x
            for downscale, upsample in ((1, 2), (1, 1), (4, 1), (2, 5), (1, 3)):
                window.downscale_spin.setValue(downscale)
                window.upsample_spin.setValue(upsample)
                self.assertEqual((window.downscale_spin.value(), window.upsample_spin.value()),
                                 (downscale, upsample))
            window.consolidate_check.setChecked(False)
            self.assertEqual((window.downscale_spin.value(), window.upsample_spin.value()), (1, 3))
        finally:
            window.worker_thread = None
            window.close()
            window.deleteLater()


    def test_without_consolidation_scale_factors_switch_instead_of_combining(self):
        settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        with patch('ditherpal.QSettings', return_value=settings):
            window = DitherApp()
        try:
            factors = lambda: (window.downscale_spin.value(), window.upsample_spin.value())
            usable = lambda: (window.restore_upscale_check.isEnabled(), window.restore_size_check.isEnabled())
            active = lambda: (window.upsample_group.property('scaleActive'),
                              window.downscale_group.property('scaleActive'))
            self.assertEqual(usable(), (True, True))
            window.downscale_spin.setValue(3)
            self.assertEqual(usable(), (False, True))
            self.assertEqual(active(), (False, True))
            # Both factors stay editable; setting one above 1x switches to it
            self.assertTrue(window.upsample_spin.isEnabled())
            window.upsample_spin.setValue(2)
            self.assertEqual(factors(), (1, 2))
            self.assertEqual(usable(), (True, False))
            self.assertEqual(active(), (True, False))
            window.downscale_spin.setValue(4)
            self.assertEqual(factors(), (4, 1))
            # Pixel Art's preset upscale doesn't shut out a solo downscale
            window.current_image_path = str(self.source)
            window.pixel_art_combo.setCurrentIndex(1)
            self.assertEqual(factors(), (1, 2))
            window.downscale_spin.setValue(2)
            self.assertEqual(factors(), (2, 1))
            # Consolidation uses both
            window.consolidate_check.setChecked(True)
            window.upsample_spin.setValue(3)
            self.assertEqual(factors(), (2, 3))
            self.assertEqual(usable(), (True, True))
            self.assertEqual(active(), (True, True))
            # Turning it off keeps the factor changed last
            window.consolidate_check.setChecked(False)
            self.assertEqual(factors(), (1, 3))
        finally:
            window.close()
            window.deleteLater()


if __name__ == '__main__':
    unittest.main()
