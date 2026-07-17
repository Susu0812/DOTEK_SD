import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from low_light_dataset.artifacts import (
    finalize_labels,
    save_label,
    validate_prepared_bundle,
)


def make_bundle(root: Path, *, reviewed: bool, hose_visible: bool = True) -> Path:
    (root / "enhanced").mkdir(parents=True)
    (root / "label").mkdir()
    stem = "lowlight_camera_full_rgb_t000000"
    Image.new("RGB", (640, 480), (50, 50, 50)).save(
        root / "enhanced" / f"{stem}.jpg"
    )
    label = np.zeros((480, 640), dtype=np.uint8)
    if hose_visible:
        label[200:480, 320] = 1
    save_label(root / "label" / f"{stem}.png", label)
    document = {
        "schema_version": 1,
        "frames": [
            {
                "stem": stem,
                "reviewed": reviewed,
                "approved": reviewed,
                "hose_visible": hose_visible,
                "final_anchors": [],
            }
        ],
    }
    (root / "annotation.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    return root


class LabelIoTests(unittest.TestCase):
    def test_saved_label_is_lossless_single_channel_binary_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "标签" / "sample.png"
            label = np.zeros((480, 640), dtype=np.uint8)
            label[220:400, 300] = 1

            save_label(path, label)

            with Image.open(path) as loaded:
                array = np.asarray(loaded)
                self.assertEqual(loaded.mode, "L")
                self.assertEqual(loaded.size, (640, 480))
                self.assertEqual(set(np.unique(array)), {0, 1})

    def test_non_binary_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            label = np.full((480, 640), 255, dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "binary"):
                save_label(Path(directory) / "bad.png", label)


class ReviewGateTests(unittest.TestCase):
    def test_finalize_applies_reviewed_anchors_and_negative_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "original").mkdir(parents=True)
            (root / "enhanced").mkdir(parents=True)
            (root / "label").mkdir()
            (root / "overlay").mkdir()
            stems = [
                "lowlight_camera_full_rgb_t000000",
                "lowlight_camera_full_rgb_t000005",
            ]
            frames = []
            for index, stem in enumerate(stems):
                Image.new("RGB", (640, 480), (40, 40, 40)).save(
                    root / "original" / f"{stem}.jpg"
                )
                Image.new("RGB", (640, 480), (50, 50, 50)).save(
                    root / "enhanced" / f"{stem}.jpg"
                )
                frames.append(
                    {
                        "stem": stem,
                        "target_timestamp_seconds": index * 5,
                        "initial_anchors": [],
                        "final_anchors": [],
                        "warnings": [],
                        "reviewed": False,
                        "approved": False,
                        "hose_visible": None,
                        "corrected": False,
                    }
                )
            (root / "annotation.json").write_text(
                json.dumps({"schema_version": 1, "frames": frames}),
                encoding="utf-8",
            )
            review = root / "review.json"
            review.write_text(
                json.dumps(
                    {
                        "decisions": {
                            stems[0]: {
                                "approved": True,
                                "hose_visible": True,
                                "anchors": [
                                    {"y": 220, "x": 300},
                                    {"y": 340, "x": 310},
                                    {"y": 479, "x": 320},
                                ],
                            },
                            stems[1]: {
                                "approved": True,
                                "hose_visible": False,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = finalize_labels(root, review, expected_count=2)

            self.assertTrue(report.ok, report.errors)
            with Image.open(root / "label" / f"{stems[0]}.png") as positive:
                self.assertGreater(int(np.asarray(positive).sum()), 0)
            with Image.open(root / "label" / f"{stems[1]}.png") as negative:
                self.assertEqual(int(np.asarray(negative).sum()), 0)

    def test_unreviewed_frame_blocks_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory), reviewed=False)

            report = validate_prepared_bundle(root, expected_count=1)

            self.assertFalse(report.ok)
            self.assertTrue(any("unreviewed" in error for error in report.errors))

    def test_same_stem_reviewed_binary_pair_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory), reviewed=True)

            report = validate_prepared_bundle(root, expected_count=1)

            self.assertTrue(report.ok, report.errors)

    def test_reviewed_negative_sample_may_have_empty_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory), reviewed=True, hose_visible=False)

            report = validate_prepared_bundle(root, expected_count=1)

            self.assertTrue(report.ok, report.errors)

    def test_visible_hose_with_empty_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory), reviewed=True, hose_visible=False)
            document_path = root / "annotation.json"
            document = json.loads(document_path.read_text(encoding="utf-8"))
            document["frames"][0]["hose_visible"] = True
            document_path.write_text(json.dumps(document), encoding="utf-8")

            report = validate_prepared_bundle(root, expected_count=1)

            self.assertFalse(report.ok)
            self.assertTrue(any("empty" in error for error in report.errors))

    def test_label_with_multiple_points_in_one_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_bundle(Path(directory), reviewed=True)
            label_path = (
                root / "label" / "lowlight_camera_full_rgb_t000000.png"
            )
            with Image.open(label_path) as image:
                label = np.asarray(image).copy()
            label[300, 321] = 1
            save_label(label_path, label)

            report = validate_prepared_bundle(root, expected_count=1)

            self.assertFalse(report.ok)
            self.assertTrue(
                any("more than one point" in error for error in report.errors)
            )


if __name__ == "__main__":
    unittest.main()
