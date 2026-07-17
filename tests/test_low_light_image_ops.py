import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from low_light_dataset.image_ops import (
    enhance_low_light,
    sample_times,
    save_jpeg,
)


class TimestampTests(unittest.TestCase):
    def test_first_five_minutes_every_five_seconds(self):
        values = sample_times()

        self.assertEqual(len(values), 60)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 295)
        self.assertTrue(all(b - a == 5 for a, b in zip(values, values[1:])))

    def test_rejects_non_positive_time_arguments(self):
        with self.assertRaises(ValueError):
            sample_times(duration_seconds=0)
        with self.assertRaises(ValueError):
            sample_times(interval_seconds=0)


class EnhancementTests(unittest.TestCase):
    def test_dark_frame_is_brightened_without_changing_contract(self):
        image = np.full((480, 640, 3), 35, dtype=np.uint8)

        result = enhance_low_light(image)

        self.assertEqual(result.image.shape, image.shape)
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertGreater(result.after.median, result.before.median)
        self.assertLessEqual(result.after.highlight_fraction, 0.01)

    def test_bright_frame_is_not_overexposed(self):
        ramp = np.tile(np.linspace(90, 220, 640, dtype=np.uint8), (480, 1))
        image = np.repeat(ramp[:, :, None], 3, axis=2)

        result = enhance_low_light(image)

        self.assertLess(result.params.retinex_weight, 0.25)
        self.assertLess(result.after.highlight_fraction, 0.02)

    def test_rejects_non_uint8_or_non_bgr_input(self):
        with self.assertRaises(TypeError):
            enhance_low_light(np.zeros((10, 10, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            enhance_low_light(np.zeros((10, 10), dtype=np.uint8))

    def test_unicode_safe_jpeg_is_training_compatible(self):
        image = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "低照度" / "样本.jpg"

            save_jpeg(path, image)

            with Image.open(path) as loaded:
                self.assertEqual(loaded.size, (640, 480))
                self.assertEqual(loaded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
