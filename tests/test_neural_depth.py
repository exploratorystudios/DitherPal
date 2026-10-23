import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEvent, QSettings
from PyQt6 import sip
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QSlider

import ditherpal
from ditherpal import DitherApp, DitherWorker
from ditherpal_depth.model import image_and_depth, infer_depth, scale_depth


class NeuralDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        Image.new('RGB', (64, 48), 'red').save(self.source)

    def tearDown(self):
        self.qt.processEvents()
        self.temp.cleanup()

    def test_strength_is_one_shared_normalized_control(self):
        depth = np.array([[0, .25, .5, .75, 1]], np.float32)
        np.testing.assert_array_equal(scale_depth(depth, 0), .5)
        np.testing.assert_array_equal(scale_depth(depth, 100), depth)
        np.testing.assert_array_equal(scale_depth(depth, 200), [[0, 0, .5, 1, 1]])
        with self.assertRaises(ValueError):
            scale_depth(depth, 201)

    def test_file_depth_cache_reuses_inference_and_invalidates_changed_source(self):
        predicted = np.linspace(0, 1, 48*64, dtype=np.float32).reshape(48, 64)
        with patch('ditherpal_depth.model.infer_depth', return_value=predicted) as infer:
            _, first = image_and_depth(self.source)
            _, second = image_and_depth(self.source)
            self.assertIs(first, second)
            self.assertFalse(first.flags.writeable)
            self.assertEqual(infer.call_count, 1)
            Image.new('RGB', (65, 48), 'blue').save(self.source)
            image_and_depth(self.source)
            self.assertEqual(infer.call_count, 2)

    def test_onnx_session_is_reused(self):
        rgb = np.zeros((16, 16, 3), np.uint8)
        with patch('ditherpal_depth.model.download_model', return_value=self.root/'model.onnx'):
            with patch('onnxruntime.InferenceSession') as session:
                session.return_value.get_inputs.return_value = [SimpleNamespace(name='pixel_values', shape=[1, 3, 14, 14])]
                session.return_value.run.return_value = [np.ones((1, 14, 14), np.float32)]
                infer_depth(rgb, self.root/'model.onnx')
                infer_depth(rgb, self.root/'model.onnx')
                self.assertEqual(session.call_count, 1)
                options = session.call_args.args[1]
                self.assertFalse(options.enable_cpu_mem_arena)
                self.assertFalse(options.enable_mem_pattern)

    def test_source_cache_stays_bounded_across_many_images(self):
        from ditherpal_depth import model
        predicted = np.linspace(0, 1, 48*64, dtype=np.float32).reshape(48, 64)
        with patch('ditherpal_depth.model.infer_depth', return_value=predicted):
            for index in range(12):
                source = self.root/f'photo-{index}.png'
                Image.new('RGB', (64, 48), 'blue').save(source)
                image_and_depth(source)
                self.assertLessEqual(len(model._files), 4)
        self.assertTrue(all('photo-' in key[0] for key in model._files))

    def test_repeated_gui_renders_release_worker_buffers_on_success_and_error(self):
        settings = QSettings(str(self.root/'memory-settings.ini'), QSettings.Format.IniFormat)
        with patch('ditherpal.QSettings', return_value=settings):
            window = DitherApp()
        window.current_image_path = str(self.source)
        window.pixel_art_combo.setCurrentIndex(1)
        near = np.full((48, 64), .7, np.float32)
        try:
            with patch('ditherpal.QMessageBox.information'), patch('ditherpal.QMessageBox.critical') as error:
                for failure in (False, False, True):
                    with patch('ditherpal.neural_depth_for_file', side_effect=RuntimeError('Model failed') if failure else None,
                               return_value=near):
                        window.process_image()
                        worker = window.worker_thread
                        deadline = time.monotonic()+15
                        while window.worker_thread is not None and time.monotonic() < deadline:
                            QTest.qWait(10)
                        self.assertIsNone(window.worker_thread)
                        self.assertTrue(window.process_btn.isEnabled())
                        self.assertIsNone(worker.source_depth)
                        self.assertIsNone(worker.preview_frame)
                        self.assertIsNone(worker.locked_palette)
                        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                        self.assertTrue(sip.isdeleted(worker))
                error.assert_called_once()
            self.assertTrue(Path(window.current_result_path).is_file())
            self.assertTrue(window.view_btn.isEnabled())
        finally:
            if window.worker_thread is not None:
                window.worker_thread.requestInterruption()
                window.worker_thread.wait()
                self.qt.processEvents()
            window.close()
            window.deleteLater()

    def test_both_pixel_art_styles_and_color_modes_receive_neural_depth(self):
        near = np.array([[0, .5], [.75, 1]], np.float32)
        for style, function in [('high_res', 'high_res_pixel_dither'), ('retro', 'retro_pixel_dither')]:
            for color_mode in ('bw', 'full_color', 'custom'):
                with self.subTest(style=style, color_mode=color_mode):
                    output = self.root/f'{style}-{color_mode}.png'
                    worker = DitherWorker(str(self.source), str(output), 'hybrid bayer + floyd-steinberg',
                                          2, 1, 1, color_mode=color_mode, custom_colors=[[0, 0, 0], [255, 255, 255]],
                                          cel_shade=True, hybrid_cell=4, pixel_art_style=style, depth_strength=0)
                    errors = []
                    worker.error.connect(errors.append)
                    def shade(*args, **kwargs):
                        return args[1] if style == 'high_res' else args[0]
                    with patch('ditherpal.neural_depth_for_file', return_value=near) as source_depth:
                        with patch('ditherpal.predict_near', side_effect=AssertionError('Inferred depth twice')):
                            with patch('ditherpal.'+function, side_effect=shade) as effect:
                                worker.run()
                    self.assertEqual(errors, [])
                    source_depth.assert_called_once()
                    passed = effect.call_args.args[3 if style == 'high_res' else 2]
                    np.testing.assert_array_equal(passed, .5)
                    self.assertTrue(output.is_file())

    def test_video_keeps_matching_neural_map_on_each_frame(self):
        source = self.root/'source.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12, (64, 48))
        self.assertTrue(writer.isOpened())
        for value in (40, 200, 80):
            writer.write(np.full((48, 64, 3), value, np.uint8))
        writer.release()
        worker = DitherWorker(str(source), str(self.root/'video.mp4'), 'hybrid bayer + floyd-steinberg',
                              2, 1, 1, color_mode='full_color', cel_shade=True, hybrid_cell=4, pixel_art_style='high_res')
        records, errors = [], []
        worker.error.connect(errors.append)
        def depth(rgb, progress=None):
            return np.full(rgb.shape[:2], float(rgb.mean())/255, np.float32)
        def dither(rgb, method, levels, depth=None):
            records.append((float(rgb.mean())/255, float(depth.mean())))
            return rgb
        with patch('ditherpal.predict_near', side_effect=depth) as infer:
            with patch.object(worker, 'apply_dithering', side_effect=dither):
                worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(infer.call_count, 3)
        self.assertEqual(len(records), 3)
        for brightness, nearness in records:
            self.assertAlmostEqual(brightness, nearness, places=5)

    def test_camera_uses_projected_depth_without_second_inference_or_strength(self):
        raw = np.full((48, 64), .8, np.float32)
        for format in ('mp4', 'gif', 'svg'):
            with self.subTest(format=format):
                output = self.root/f'zoom.{format}'
                worker = DitherWorker(str(self.source), str(output), 'hybrid bayer + floyd-steinberg',
                                      2, 1, 1, color_mode='full_color', cel_shade=True, hybrid_cell=4,
                                      pixel_art_style='retro', zoom_format=format, depth_strength=0)
                errors = []
                worker.error.connect(errors.append)
                frames = [(np.full((48, 64, 3), 70, np.uint8), np.full((48, 64), n, np.float32)) for n in (.2, .7)]
                with patch('ditherpal.neural_depth_for_file', return_value=raw):
                    with patch('ditherpal.NeuralScene') as scene:
                        viewer = self.root/f'viewer-{format}.html'
                        viewer.write_text('<html>mesh fixture</html>')
                        scene.return_value.write.return_value = str(viewer)
                        scene.return_value.frames.return_value = iter(frames)
                        with patch('ditherpal.retro_pixel_dither', side_effect=lambda art, *args: art) as effect:
                            with patch('ditherpal.ZOOM_FRAMES', 2):
                                with patch('ditherpal.predict_near', side_effect=AssertionError('Repeated camera inference')):
                                    worker.run()
                self.assertEqual(errors, [])
                self.assertEqual(scene.call_args.args[2], 0)
                self.assertTrue(output.is_file())
                np.testing.assert_allclose(effect.call_args_list[0].args[2], .2)
                np.testing.assert_allclose(effect.call_args_list[1].args[2], .7)

    def test_model_failure_is_explicit_without_fallback(self):
        worker = DitherWorker(str(self.source), str(self.root/'fail.png'), 'hybrid bayer + floyd-steinberg',
                              2, 1, 1, cel_shade=True, pixel_art_style='retro')
        errors = []
        worker.error.connect(errors.append)
        with patch('ditherpal.neural_depth_for_file', side_effect=RuntimeError('Model unavailable')):
            worker.run()
        self.assertEqual(len(errors), 1)
        self.assertIn('Model unavailable', errors[0])
        self.assertFalse((self.root/'fail.png').exists())
        self.assertFalse(hasattr(ditherpal, 'estimate_depth'))
        self.assertFalse(hasattr(ditherpal, 'zoom_frame'))

    def test_popup_has_one_slider_and_updates_all_pipeline_settings(self):
        settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)
        with patch('ditherpal.QSettings', return_value=settings):
            window = DitherApp()
        try:
            window.current_image_path = str(self.source)
            window.update_depth_controls()
            self.assertFalse(hasattr(window, 'zoom_engine_combo'))
            rgb = np.zeros((48, 64, 3), np.uint8)
            near = np.full((48, 64), .8, np.float32)
            with patch('depth_editor.image_and_depth', return_value=(rgb, near)) as infer:
                window.depth_map_btn.click()
                dialog = window.depth_dialog
                deadline = time.monotonic()+5
                while dialog.job is not None and time.monotonic()<deadline:
                    QTest.qWait(10)
                self.assertIsNotNone(dialog.raw_depth)
                self.assertEqual(len(dialog.findChildren(QSlider)), 1)
                dialog.slider.setValue(0)
                self.assertEqual(window.depth_strength, 0)
                dialog.slider.setValue(140)
                self.assertEqual(window.depth_strength, 140)
                infer.assert_called_once()
                dialog.close()
                self.qt.processEvents()
            for format in (0, 2):
                window.zoom_combo.setCurrentIndex(format)
                with patch('ditherpal.DitherWorker') as worker:
                    window.process_image()
                self.assertEqual(worker.call_args.kwargs['depth_strength'], 140)
                window.worker_thread = None
            settings.sync()
            self.assertEqual(settings.value('depth_strength', type=int), 140)
        finally:
            window.worker_thread = None
            if window.depth_dialog is not None:
                window.depth_dialog.close()
            window.close()
            window.deleteLater()


if __name__ == '__main__':
    unittest.main()
