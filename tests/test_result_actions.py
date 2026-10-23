"""Run with QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QListWidget

from ditherpal import DitherApp, ComparisonWorker


class ResultActionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self.settings_patch = patch("ditherpal.QSettings", return_value=self.settings)
        self.settings_patch.start()
        self.window = DitherApp()
        self.window.current_image_path = str(self.root / "source.png")
        self.outputs = self.root / "dithered"
        self.outputs.mkdir()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.processEvents()
        self.settings_patch.stop()
        self.temp.cleanup()

    def image(self, name, color):
        path = self.outputs / name
        Image.new("RGB", (32, 24), color).save(path)
        return path

    def test_completion_persists_history_and_enables_actions(self):
        self.assertFalse(self.window.view_btn.isEnabled())
        self.assertFalse(self.window.finalize_btn.isEnabled())
        path = self.image("red.png", "red")
        with patch("ditherpal.QMessageBox.information"):
            self.window.on_finished("Saved", str(path))
        self.assertEqual(self.window.current_result_path, str(path))
        self.assertTrue(self.window.view_btn.isEnabled())
        self.assertTrue(self.window.finalize_btn.isEnabled())
        self.settings.sync()
        fresh = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self.assertEqual(fresh.value("recent_results", [], type=list), [str(path)])
        self.window.remember_result(path)
        self.assertEqual(self.window.recent_results, [str(path)])

    def test_recent_click_changes_preview_and_view_finalize_target(self):
        red = self.image("red.png", "red")
        blue = self.image("blue.png", "blue")
        self.window.show_after_preview(red)
        self.window.show_recent_results()
        dialog = self.window.findChildren(QDialog)[0]
        results = dialog.findChild(QListWidget)
        self.assertEqual(results.count(), 2)
        for row in range(results.count()):
            if results.item(row).text().startswith("blue.png"):
                results.setCurrentRow(row)
        self.assertEqual(self.window.current_result_path, str(blue))
        pixel = self.window.after_preview.pixmap().toImage().pixelColor(0, 0)
        self.assertEqual(pixel.name(), "#0000ff")
        self.assertEqual(self.window.current_image_path, str(self.root / "source.png"))
        with patch("ditherpal.shutil.which", return_value=None):
            with patch("ditherpal.QDesktopServices.openUrl", return_value=True) as opened:
                self.window.view_btn.click()
                self.assertEqual(opened.call_args.args[0].toLocalFile(), str(blue))
        self.window.finalize_btn.click()
        self.assertEqual((self.outputs / "Finalized" / blue.name).read_bytes(), blue.read_bytes())
        dialog.close()

    def test_finalize_preserves_source_and_existing_favorites(self):
        path = self.image("favorite.png", "red")
        original = path.read_bytes()
        self.window.show_after_preview(path)
        self.window.finalize_result()
        self.window.finalize_result()
        folder = self.outputs / "Finalized"
        self.assertEqual((folder / "favorite.png").read_bytes(), original)
        self.assertEqual((folder / "favorite_1.png").read_bytes(), original)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.window.recent_result_paths(), [path])

    def wait_for_comparison(self):
        deadline = time.monotonic() + 5
        while self.window.comparison_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.comparison_worker)

    def test_comparison_uses_recorded_original_and_preserves_existing_exports(self):
        source = self.root / "original.png"
        Image.new("RGB", (64, 48), "red").save(source)
        result = self.image("blue.png", "blue")
        self.window.remember_result(result, source)
        self.window.current_image_path = str(self.root / "another-original.png")
        self.window.show_after_preview(result)
        with patch("ditherpal.QFileDialog.getOpenFileName") as choose:
            self.window.comparison_btn.click()
            self.wait_for_comparison()
            choose.assert_not_called()
        folder = self.outputs / "Comparisons"
        comparison = folder / "blue_comparison.png"
        original_export = comparison.read_bytes()
        with Image.open(comparison) as image:
            # Native dither is 32x24; each panel has a 192px minimum for labels.
            self.assertEqual(image.size, (420, 72))
            self.assertEqual(image.getpixel((108, 48)), (255, 0, 0))
            self.assertEqual(image.getpixel((312, 48)), (0, 0, 255))
        self.window.comparison_btn.click()
        self.wait_for_comparison()
        self.assertTrue((folder / "blue_comparison_1.png").is_file())
        self.assertEqual(comparison.read_bytes(), original_export)
        self.assertEqual(self.window.current_result_path, str(result))
        self.assertEqual(self.window.recent_result_paths(), [result])
        self.settings.sync()
        fresh = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        self.assertEqual(fresh.value("result_sources", {}, type=dict)[str(result)], str(source))

    def test_comparison_recognizes_older_output_for_current_source(self):
        source = self.root / "source.png"
        Image.new("RGB", (64, 48), "red").save(source)
        result = self.image("source_dithered_fullcolor_3.png", "blue")
        self.window.show_after_preview(result)
        with patch("ditherpal.QFileDialog.getOpenFileName") as choose:
            self.window.save_comparison()
            self.wait_for_comparison()
        choose.assert_not_called()
        self.assertEqual(self.window.result_sources[str(result)], str(source))

    def test_unknown_original_is_requested_and_cancel_does_not_export(self):
        source = self.root / "source.png"
        Image.new("RGB", (64, 48), "red").save(source)
        result = self.image("source_dithered_other_dithered_bw.png", "blue")
        self.window.show_after_preview(result)
        with patch("ditherpal.QFileDialog.getOpenFileName", return_value=("", "")) as choose:
            self.window.save_comparison()
        choose.assert_called_once()
        self.assertIsNone(self.window.comparison_worker)
        self.assertFalse((self.outputs / "Comparisons").exists())

    def test_failed_comparison_removes_partial_export(self):
        source = self.image("red.png", "red")
        result = self.image("blue.png", "blue")
        worker = ComparisonWorker(source, result)
        errors = []
        worker.error.connect(errors.append)
        with patch("PIL.Image.Image.save", side_effect=OSError("Disk full")):
            worker.run()
        self.assertEqual(errors, ["Disk full"])
        self.assertEqual(list((self.outputs / "Comparisons").iterdir()), [])

    def test_view_uses_async_gio_without_stale_startup_tokens(self):
        path = self.image("version with spaces.png", "red")
        self.window.show_after_preview(path)
        with patch("ditherpal.shutil.which", return_value="/usr/bin/gio"):
            with patch("ditherpal.QProcess") as process_class:
                with patch.dict(os.environ, {"DESKTOP_STARTUP_ID": "old-startup", "XDG_ACTIVATION_TOKEN": "old-token"}):
                    self.window.view_result()
                process = process_class.return_value
                process.start.assert_called_once_with("/usr/bin/gio", ["open", str(path)])
                environment = process.setProcessEnvironment.call_args.args[0]
                self.assertFalse(environment.contains("DESKTOP_STARTUP_ID"))
                self.assertFalse(environment.contains("XDG_ACTIVATION_TOKEN"))
                process.waitForStarted.assert_not_called()
                process.waitForFinished.assert_not_called()
                self.window.viewer_launches.clear()

    def run_fake_viewer(self, body):
        opener = self.root / "fake-gio"
        opener.write_text("#!/usr/bin/python3\n" + body)
        opener.chmod(0o755)
        path = self.image("red.png", "red")
        self.window.show_after_preview(path)
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(bool(self.window.viewer_launches)))
        with patch("ditherpal.shutil.which", return_value=str(opener)):
            self.window.view_result()
        deadline = time.monotonic() + 3
        while self.window.viewer_launches and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertEqual(heartbeat, [True], "GUI must handle events while the launcher is still running")
        self.assertFalse(self.window.viewer_launches)

    def test_slow_viewer_keeps_gui_responsive(self):
        self.run_fake_viewer("import time\ntime.sleep(0.2)\n")
        self.assertIn("Sent to viewer", self.window.status_label.text())

    def test_viewer_failure_is_reported(self):
        with patch("ditherpal.QMessageBox.warning") as warning:
            self.run_fake_viewer("import sys\nsys.stderr.write('No viewer available')\nsys.exit(2)\n")
        self.assertEqual(warning.call_args.args[2], "No viewer available")

    def test_missing_result_disables_actions_and_is_omitted(self):
        path = self.image("deleted.png", "red")
        self.window.remember_result(path)
        self.window.show_after_preview(path)
        path.unlink()
        with patch("ditherpal.QMessageBox.warning") as warning:
            self.window.view_result()
            warning.assert_called_once()
        self.assertFalse(self.window.view_btn.isEnabled())
        self.assertFalse(self.window.finalize_btn.isEnabled())
        self.assertEqual(self.window.recent_result_paths(), [])

    def test_failed_copy_removes_partial_favorite(self):
        path = self.image("favorite.png", "red")
        self.window.show_after_preview(path)
        with patch("ditherpal.shutil.copyfileobj", side_effect=OSError("Disk full")):
            with patch("ditherpal.QMessageBox.warning") as warning:
                self.window.finalize_result()
                warning.assert_called_once()
        self.assertEqual(list((self.outputs / "Finalized").iterdir()), [])
        self.assertTrue(path.is_file())

    def test_history_is_bounded_and_newest_first(self):
        for index in range(105):
            self.window.remember_result(self.outputs / f"result_{index}.png")
        self.assertEqual(len(self.window.recent_results), 100)
        self.assertTrue(self.window.recent_results[0].endswith("result_104.png"))
        self.assertTrue(self.window.recent_results[-1].endswith("result_5.png"))

    def test_video_output_format_gets_unique_filename_before_processing(self):
        self.window.current_image_path = str(self.root / "source.avi")
        existing = self.outputs / "source_dithered_bw.gif"
        existing.write_bytes(b"existing output")
        self.window.video_format_combo.setCurrentIndex(1)
        self.window.video_palette_check.setChecked(False)
        with patch("ditherpal.DitherWorker") as worker:
            self.window.process_image()
        args = worker.call_args.args
        self.assertEqual(args[1], str(self.outputs / "source_dithered_bw_1.gif"))
        self.assertEqual(args[-2:], ("gif", False))
        self.assertEqual(existing.read_bytes(), b"existing output")

    def test_svg_reloads_without_worker(self):
        path = self.outputs / "animation.svg"
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="24">'
                        '<rect width="32" height="24" fill="red"/></svg>')
        self.window.show_after_preview(path)
        self.assertEqual(self.window.after_preview.pixmap().toImage().pixelColor(0, 0).name(), "#ff0000")

    def test_video_reloads_its_own_frame_without_worker(self):
        import cv2
        import numpy as np

        for name, color in (("red.mp4", (0, 0, 255)), ("blue.mp4", (255, 0, 0))):
            path = self.outputs / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
            finally:
                writer.release()
            self.window.show_after_preview(path)
            pixel = self.window.after_preview.pixmap().toImage().pixelColor(0, 0)
            if name == "red.mp4":
                self.assertGreater(pixel.red(), 240)
                self.assertLess(pixel.blue(), 10)
            else:
                self.assertGreater(pixel.blue(), 240)
                self.assertLess(pixel.red(), 10)


if __name__ == "__main__":
    unittest.main()
