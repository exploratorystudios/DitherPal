import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from ditherpal import DitherWorker, StreamingGifWriter, SKLEARN_AVAILABLE


class VideoPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def worker(self, **kwargs):
        return DitherWorker("", str(self.root / "output.mp4"),
                            "hybrid bayer + floyd-steinberg", 8, 1, 1,
                            color_mode="full_color", **kwargs)

    def source_video(self):
        path = self.root / "source.mp4"
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
        self.assertTrue(video.isOpened())
        try:
            for index in range(10):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :32] = (0, 0, index * 25)
                frame[:, 32:] = (255 - index * 25, 255, 0)
                video.write(frame)
        finally:
            video.release()
        return path

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
    def test_grid_cel_shading_skips_unused_full_color_palette(self):
        worker = self.worker(cel_shade=True, hybrid_cell=4)
        rgb = np.random.default_rng(1).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        with patch("sklearn.cluster.KMeans") as clustering:
            rendered = worker.apply_dithering(rgb, worker.method, worker.levels)
        clustering.assert_not_called()
        expected = worker._apply_hybrid_grid(rgb.astype(np.float32), None)
        np.testing.assert_array_equal(rendered, expected)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is optional")
    def test_locked_palette_is_built_once_before_parallel_work(self):
        worker = self.worker()
        worker.method = "floyd-steinberg"
        worker.lock_palette = True
        frames = [np.full((40, 40, 3), color, dtype=np.uint8) for color in (32, 64, 128, 192)]
        with patch("sklearn.cluster.KMeans") as clustering:
            clustering.return_value.cluster_centers_ = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32)
            with patch("ditherpal.floyd_steinberg_color_numba", side_effect=lambda image, palette: image.astype(np.uint8)):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    worker.frame_executor = executor
                    first = worker.process_frame_batch(frames[:2])
                    second = worker.process_frame_batch(frames[2:])
        self.assertEqual(clustering.return_value.fit.call_count, 1)
        self.assertTrue(worker.video_palette_seeded)
        for source, result in zip(frames, first + second):
            np.testing.assert_array_equal(source, result)

    def test_streamed_gif_preserves_frames_colors_and_timing(self):
        path = self.root / "stream.gif"
        writer = StreamingGifWriter(path, 30)
        colors = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]
        try:
            for index in range(30):
                writer.write(np.full((8, 8, 3), colors[index % 3], dtype=np.uint8))
        finally:
            writer.release()
        writer.release()  # Closing twice must be harmless.
        with Image.open(path) as gif:
            self.assertEqual(gif.n_frames, 30)
            self.assertEqual(gif.info["loop"], 0)
            durations = []
            for index in range(gif.n_frames):
                gif.seek(index)
                durations.append(gif.info["duration"])
                self.assertEqual(gif.convert("RGB").getpixel((0, 0)), colors[index % 3])
            self.assertEqual(sum(durations), 1000)

    def test_video_exports_mp4_and_gif_with_one_pool_and_all_frames(self):
        source = self.source_video()
        original = source.read_bytes()
        for format in ("mp4", "gif"):
            with self.subTest(format=format):
                output = self.root / f"result.{format}"
                worker = DitherWorker(str(source), str(output), "bayer 2x2", 2, 1, 1,
                                      color_mode="bw", video_format=format)
                errors, completions = [], []
                worker.error.connect(errors.append)
                worker.finished.connect(lambda message, path: completions.append(path))
                with patch("ditherpal.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool:
                    worker.run()
                self.assertEqual(errors, [])
                self.assertEqual(completions, [str(output)])
                self.assertEqual(pool.call_count, 1)
                self.assertIsNone(worker.frame_executor)
                self.assertFalse(worker.lock_palette)
                self.assertIsNotNone(worker.preview_frame)
                if format == "gif":
                    with Image.open(output) as gif:
                        self.assertEqual(gif.n_frames, 10)
                        self.assertEqual(gif.size, (64, 48))
                else:
                    cap = cv2.VideoCapture(str(output))
                    count = 0
                    try:
                        while cap.read()[0]:
                            count += 1
                    finally:
                        cap.release()
                    self.assertEqual(count, 10)
        self.assertEqual(source.read_bytes(), original)

    def test_failure_releases_capture(self):
        worker = self.worker()
        errors = []
        worker.error.connect(errors.append)
        with patch("cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = False
            worker.process_video("missing.mp4")
            capture.return_value.release.assert_called_once()
        self.assertEqual(len(errors), 1)
        self.assertFalse(worker.lock_palette)


if __name__ == "__main__":
    unittest.main()
