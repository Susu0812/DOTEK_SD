import unittest

import numpy as np

from low_light_dataset.annotation import (
    AnchorPrediction,
    decode_logits,
    extract_color_anchors,
    fuse_predictions,
    rasterize_centerline,
    regularize_anchors,
)


class LogitDecodeTests(unittest.TestCase):
    def test_decodes_51_by_18_by_1_logits(self):
        logits = np.full((1, 51, 18, 1), -8.0, dtype=np.float32)
        logits[:, 24, :, :] = 8.0

        points = decode_logits(logits, width=640, height=480)

        self.assertEqual(len(points), 18)
        self.assertTrue(all(point.confidence > 0.99 for point in points))
        self.assertAlmostEqual(points[0].x, 24.0 * 639.0 / 49.0, places=2)
        self.assertEqual(points[0].y, round(121 * 479 / 287))

    def test_rejects_wrong_model_shape(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            decode_logits(np.zeros((1, 51, 17, 1), dtype=np.float32), 640, 480)


class PredictionFusionTests(unittest.TestCase):
    def test_color_fallback_tracks_center_of_red_hose(self):
        image = np.full((480, 640, 3), 45, dtype=np.uint8)
        for y in range(190, 480):
            center = 250 + (y - 190) // 4
            image[y, center - 24 : center + 25] = (25, 35, 145)

        points = extract_color_anchors(image)

        self.assertGreaterEqual(len(points), 12)
        bottom = max(points, key=lambda point: point.y)
        expected = 250 + (bottom.y - 190) // 4
        self.assertAlmostEqual(bottom.x, expected, delta=3)
        self.assertTrue(all(point.source == "color" for point in points))

    def test_fusion_weights_agreeing_points_by_confidence(self):
        original = [AnchorPrediction(300, 100.0, 0.7, "original")]
        enhanced = [AnchorPrediction(300, 110.0, 0.9, "enhanced")]

        fused, warnings = fuse_predictions(original, enhanced)

        self.assertAlmostEqual(fused[0].x, 105.625, places=3)
        self.assertEqual(fused[0].source, "fused")
        self.assertEqual(warnings, [])

    def test_fusion_flags_large_disagreement_and_uses_higher_confidence(self):
        original = [AnchorPrediction(300, 100.0, 0.6, "original")]
        enhanced = [AnchorPrediction(300, 220.0, 0.9, "enhanced")]

        fused, warnings = fuse_predictions(original, enhanced)

        self.assertEqual(fused[0].x, 220.0)
        self.assertIn("prediction_disagreement_y_300", warnings)

    def test_regularization_removes_isolated_outlier(self):
        values = [
            AnchorPrediction(220, 100.0, 0.9, "fused"),
            AnchorPrediction(240, 110.0, 0.9, "fused"),
            AnchorPrediction(260, 500.0, 0.8, "fused"),
            AnchorPrediction(280, 130.0, 0.9, "fused"),
            AnchorPrediction(300, 140.0, 0.9, "fused"),
        ]

        cleaned, warnings = regularize_anchors(values)

        self.assertNotIn(500, [round(item.x) for item in cleaned])
        self.assertIn("removed_outlier_y_260", warnings)


class CenterlineTests(unittest.TestCase):
    def test_rasterized_label_is_binary_and_at_most_one_pixel_per_row(self):
        anchors = [
            AnchorPrediction(220, 100.0, 0.9, "fused"),
            AnchorPrediction(300, 180.0, 0.9, "fused"),
            AnchorPrediction(440, 260.0, 0.9, "fused"),
        ]

        label = rasterize_centerline(anchors)

        self.assertEqual(label.shape, (480, 640))
        self.assertEqual(label.dtype, np.uint8)
        self.assertEqual(set(np.unique(label)), {0, 1})
        self.assertLessEqual(int(label.sum(axis=1).max()), 1)
        self.assertEqual(int(label.sum()), 221)

    def test_rasterization_requires_three_points(self):
        anchors = [
            AnchorPrediction(220, 100.0, 0.9, "fused"),
            AnchorPrediction(300, 180.0, 0.9, "fused"),
        ]
        with self.assertRaisesRegex(ValueError, "three"):
            rasterize_centerline(anchors)


if __name__ == "__main__":
    unittest.main()
