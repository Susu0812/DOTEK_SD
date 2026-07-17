import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from low_light_dataset.artifacts import save_label
from low_light_dataset.dataset_merge import (
    merge_bundle,
    preflight_merge,
    snapshot_dataset,
)


def save_pair(root: Path, split: str, stem: str, color: int) -> None:
    image_dir = root / split / "pic"
    label_dir = root / split / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (color, color, color)).save(
        image_dir / f"{stem}.jpg",
        quality=95,
    )
    label = np.zeros((480, 640), dtype=np.uint8)
    label[220:480, 320] = 1
    save_label(label_dir / f"{stem}.png", label)


def make_dataset(root: Path) -> Path:
    dataset = root / "newdata"
    save_pair(dataset, "train", "existing_train_a", 40)
    save_pair(dataset, "train", "existing_train_b", 50)
    save_pair(dataset, "test", "existing_test", 60)
    return dataset


def make_bundle(root: Path, stem: str = "lowlight_camera_full_rgb_t000000", color: int = 80) -> Path:
    bundle = root / "bundle"
    (bundle / "enhanced").mkdir(parents=True)
    (bundle / "label").mkdir()
    Image.new("RGB", (640, 480), (color, color, color)).save(
        bundle / "enhanced" / f"{stem}.jpg",
        quality=95,
    )
    label = np.zeros((480, 640), dtype=np.uint8)
    label[220:480, 320] = 1
    save_label(bundle / "label" / f"{stem}.png", label)
    (bundle / "annotation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [
                    {
                        "stem": stem,
                        "reviewed": True,
                        "approved": True,
                        "hose_visible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return bundle


class MergePreflightTests(unittest.TestCase):
    def test_same_name_blocks_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            bundle = make_bundle(root, stem="existing_train_a")

            with self.assertRaisesRegex(ValueError, "name collision"):
                preflight_merge(
                    bundle,
                    dataset,
                    expected_new_count=1,
                    expected_train_before=2,
                    expected_test_count=1,
                )

    def test_test_image_hash_blocks_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            bundle = make_bundle(root, color=60)

            with self.assertRaisesRegex(ValueError, "test-set leakage"):
                preflight_merge(
                    bundle,
                    dataset,
                    expected_new_count=1,
                    expected_train_before=2,
                    expected_test_count=1,
                )


class MergeExecutionTests(unittest.TestCase):
    def test_successful_merge_preserves_pairing_and_test_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            bundle = make_bundle(root)

            receipt = merge_bundle(
                bundle,
                dataset,
                expected_new_count=1,
                expected_train_before=2,
                expected_test_count=1,
            )

            snapshot = snapshot_dataset(dataset)
            self.assertEqual(len(snapshot.train.images), 3)
            self.assertEqual(len(snapshot.train.labels), 3)
            self.assertEqual(set(snapshot.train.images), set(snapshot.train.labels))
            self.assertEqual(len(snapshot.test.images), 1)
            self.assertTrue(receipt.is_file())


if __name__ == "__main__":
    unittest.main()
