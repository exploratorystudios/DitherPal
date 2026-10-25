import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from ditherpal import DitherApp, DitherWorker


class NativePixelOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        rgb = np.random.default_rng(4).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(self.source)

    def worker(self, output, **kwargs):
        settings = dict(color_mode='full_color', cel_shade=True, hybrid_cell=4,
                        pixel_art_style='retro', native_pixel_size=True)
        settings.update(kwargs)
        return DitherWorker(str(self.source), str(output), 'hybrid bayer + floyd-steinberg',
                            2, settings.pop('upsample', 1), settings.pop('downscale', 1), **settings)

    def run_worker(self, worker):
        errors, completed = [], []
        worker.error.connect(errors.append)
        worker.finished.connect(lambda *args: completed.append(args))
        with patch('ditherpal.neural_depth_for_file', return_value=np.full((48, 64), .5, np.float32)):
            with patch('ditherpal.predict_near', return_value=np.full((48, 64), .5, np.float32)):
                worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(completed), 1)

    def test_native_output_is_exactly_the_existing_art_grid_in_both_modes(self):
        for style in ('retro', 'high_res'):
            with self.subTest(style=style):
                expanded = self.root/f'{style}-expanded.png'
                native = self.root/f'{style}-native.png'
                self.run_worker(self.worker(expanded, pixel_art_style=style, upsample=2, native_pixel_size=False))
                self.run_worker(self.worker(native, pixel_art_style=style, upsample=2))
                with Image.open(expanded) as full, Image.open(native) as small:
                    self.assertEqual(small.size, (32, 24))
                    np.testing.assert_array_equal(np.array(small), np.array(full)[::4, ::4])

    def test_restoration_options_cannot_expand_native_output_or_change_custom_palette(self):
        palette = [(0, 0, 0), (255, 180, 60), (70, 180, 240)]
        output = self.root/'palette.png'
        self.run_worker(self.worker(output, color_mode='custom', custom_colors=palette,
                                    upsample=3, downscale=2, restore_size=True, restore_upscale=True))
        with Image.open(output) as image:
            self.assertEqual(image.size, (24, 18))
            for color in np.unique(np.array(image).reshape(-1, 3), axis=0):
                self.assertIn(tuple(color), palette)

    def test_large_blocks_and_partial_edges_keep_every_cell(self):
        Image.new('RGB', (73, 51)).save(self.source)
        worker = self.worker(self.root/'edges.png', hybrid_cell=13, upsample=3, downscale=2,
                             restore_upscale=True)

        def chart(image, *_args, **_kwargs):
            block = worker.hybrid_cell
            y, x = np.indices((image.height, image.width))
            return np.stack([x//block*20, y//block*20, np.zeros_like(x)], axis=2).astype(np.uint8)

        with patch.object(worker, 'apply_dithering', side_effect=chart):
            self.run_worker(worker)
        with Image.open(worker.output_path) as image:
            self.assertEqual(image.size, (9, 6))
            rgb = np.array(image)
            np.testing.assert_array_equal(rgb[0, :, 0], np.arange(9)*20)
            np.testing.assert_array_equal(rgb[:, 0, 1], np.arange(6)*20)

    def test_option_is_ignored_outside_pixel_art_modes(self):
        worker = self.worker(self.root/'off.png', pixel_art_style='off')
        self.run_worker(worker)
        with Image.open(worker.output_path) as image:
            self.assertEqual(image.size, (64, 48))

    def make_video(self):
        path = self.root/'source.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for color in (30, 90, 160):
                writer.write(np.full((48, 64, 3), color, np.uint8))
        finally:
            writer.release()
        return path

    def test_video_preserves_grid_and_pads_mp4_without_resampling(self):
        source = self.make_video()
        for format in ('gif', 'mp4'):
            worker = self.worker(self.root/f'video.{format}', hybrid_cell=5, video_format=format)
            worker.input_path = str(source)
            if format == 'gif':
                self.run_worker(worker)
                with Image.open(worker.output_path) as image:
                    self.assertEqual(image.size, (13, 10))
                    self.assertEqual(image.n_frames, 3)
            else:
                with patch('cv2.VideoWriter') as writer:
                    with patch.object(worker, '_pad_video_frame', wraps=worker._pad_video_frame) as pad:
                        self.run_worker(worker)
                self.assertEqual(writer.call_args.args[-1], (14, 10))
                self.assertEqual(writer.return_value.write.call_count, 3)
                for call, written in zip(pad.call_args_list, writer.return_value.write.call_args_list):
                    grid = call.args[0]
                    frame = written.args[0][..., ::-1]
                    self.assertEqual(grid.shape, (10, 13, 3))
                    np.testing.assert_array_equal(frame[:, :13], grid)
                    np.testing.assert_array_equal(frame[:, 13], grid[:, -1])

    def test_camera_uses_native_texture_and_canvas_and_pads_only_mp4(self):
        for format in ('gif', 'mp4', 'svg'):
            with self.subTest(format=format):
                output = self.root/f'camera.{format}'
                worker = self.worker(output, hybrid_cell=5, zoom_format=format, neural_fill_holes=False)
                with patch('ditherpal.ZOOM_FRAMES', 2):
                    self.run_worker(worker)
                with Image.open(Path(worker.mesh_viewer_path).parent/'texture.png') as texture:
                    self.assertEqual(texture.size, (13, 10))
                if format == 'gif':
                    with Image.open(output) as image:
                        self.assertEqual(image.size, (13, 10))
                elif format == 'mp4':
                    cap = cv2.VideoCapture(str(output))
                    try:
                        ok, frame = cap.read()
                        self.assertTrue(ok)
                        self.assertEqual(frame.shape, (10, 14, 3))
                    finally:
                        cap.release()
                else:
                    self.assertIn('viewBox="0 0 13 10"', output.read_text())

    def test_gui_persists_setting_previews_dimensions_and_passes_it_to_worker(self):
        settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        settings.setValue('native_pixel_size', True)
        with patch('ditherpal.QSettings', return_value=settings):
            window = DitherApp()
        try:
            self.assertTrue(window.native_pixel_check.isChecked())
            self.assertFalse(window.native_pixel_check.isEnabled())
            window.pixel_art_combo.setCurrentIndex(2)
            window.current_image_path = str(self.source)
            window.upsample_spin.setValue(2)
            window.hybrid_cell_spin.setValue(4)
            window.update_dimensions_text()
            self.assertTrue(window.native_pixel_check.isEnabled())
            self.assertFalse(window.restore_size_check.isEnabled())
            self.assertFalse(window.restore_upscale_check.isEnabled())
            self.assertIn('Output: 32x24 px', window.image_label.text())
            jpeg = self.root/'input.jpg'
            Image.new('RGB', (64, 48)).save(jpeg)
            window.current_image_path = str(jpeg)
            with patch('ditherpal.DitherWorker') as worker:
                window.process_image()
            self.assertTrue(worker.call_args.kwargs['native_pixel_size'])
            self.assertEqual(Path(worker.call_args.args[1]).suffix, '.png')
            window.worker_thread = None
            window.current_image_path = str(self.make_video())
            window.upsample_spin.setValue(1)
            window.hybrid_cell_spin.setValue(5)
            self.assertIn('Output: 13x10 px', window.image_label.text())
            self.assertIn('MP4 canvas: 14x10 px', window.image_label.text())
            window.video_format_combo.setCurrentIndex(1)
            self.assertNotIn('MP4 canvas', window.image_label.text())
            window.native_pixel_check.setChecked(False)
            self.assertFalse(settings.value('native_pixel_size', type=bool))
            self.assertTrue(window.restore_size_check.isEnabled())
            self.assertTrue(window.restore_upscale_check.isEnabled())
        finally:
            window.worker_thread = None
            window.close()
            window.deleteLater()


if __name__ == '__main__':
    unittest.main()
