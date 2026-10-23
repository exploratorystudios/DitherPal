"""Run with QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QDialog, QListWidget

from ditherpal import DitherApp


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
