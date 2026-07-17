import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from low_light_dataset.quarantine import quarantine_merged_bundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def save_pair(dataset: Path, stem: str, color: int) -> tuple[Path, Path]:
    image = dataset / "train" / "pic" / f"{stem}.jpg"
    label = dataset / "train" / "label" / f"{stem}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (color, color, color)).save(image)
    Image.new("L", (32, 24), 1).save(label)
    return image, label


def make_case(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "newdata"
    save_pair(dataset, "existing", 20)
    image, label = save_pair(dataset, "lowlight_camera_full_rgb_t000000", 80)
    test_image = dataset / "test" / "pic" / "test.jpg"
    test_label = dataset / "test" / "label" / "test.png"
    test_image.parent.mkdir(parents=True)
    test_label.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 40, 40)).save(test_image)
    Image.new("L", (32, 24), 1).save(test_label)

    bundle = root / "bundle"
    bundle.mkdir()
    receipt = {
        "schema_version": 1,
        "dataset_root": str(dataset.resolve()),
        "before_counts": {"train_images": 1, "train_labels": 1,
                          "test_images": 1, "test_labels": 1},
        "after_counts": {"train_images": 2, "train_labels": 2,
                         "test_images": 1, "test_labels": 1},
        "new_files": [
            {"path": str(image.resolve()), "sha256": sha256(image)},
            {"path": str(label.resolve()), "sha256": sha256(label)},
        ],
    }
    (bundle / "merge_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return bundle, dataset, root / "quarantine"


class QuarantineTests(unittest.TestCase):
    def test_hash_mismatch_refuses_without_moving_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, dataset, quarantine = make_case(Path(directory))
            receipt = json.loads((bundle / "merge_receipt.json").read_text())
            receipt["new_files"][0]["sha256"] = "0" * 64
            (bundle / "merge_receipt.json").write_text(json.dumps(receipt))

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                quarantine_merged_bundle(
                    bundle, dataset, quarantine,
                    expected_count=1, expected_train_after=1,
                    expected_test_count=1,
                )

            self.assertTrue(
                (dataset / "train" / "pic" /
                 "lowlight_camera_full_rgb_t000000.jpg").is_file()
            )
            self.assertFalse(quarantine.exists())

    def test_success_is_paired_counted_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, dataset, quarantine = make_case(Path(directory))

            receipt = quarantine_merged_bundle(
                bundle, dataset, quarantine,
                expected_count=1, expected_train_after=1,
                expected_test_count=1,
            )

            self.assertTrue(receipt.is_file())
            self.assertEqual(len(list((dataset / "train" / "pic").glob("*.jpg"))), 1)
            self.assertEqual(len(list((dataset / "train" / "label").glob("*.png"))), 1)
            self.assertTrue(
                (quarantine / "train_pic" /
                 "lowlight_camera_full_rgb_t000000.jpg").is_file()
            )
            self.assertTrue(
                (quarantine / "train_label" /
                 "lowlight_camera_full_rgb_t000000.png").is_file()
            )
            again = quarantine_merged_bundle(
                bundle, dataset, quarantine,
                expected_count=1, expected_train_after=1,
                expected_test_count=1,
            )
            self.assertEqual(receipt, again)


if __name__ == "__main__":
    unittest.main()

