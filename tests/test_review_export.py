import csv
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from low_light_dataset.annotation import AnchorPrediction, rasterize_centerline
from low_light_dataset.artifacts import save_label
from low_light_dataset.artifacts import load_bgr
from low_light_dataset.dataset_merge import snapshot_dataset
from low_light_dataset.review_store import ReviewStore


_module_available = importlib.util.find_spec("low_light_dataset.review_export")
if _module_available is not None:
    import low_light_dataset.review_export as review_export_module

    from low_light_dataset.review_export import (
        ReviewExportError,
        assign_second_review_requirements,
        export_reviewed_bundle,
        validate_review,
    )
else:
    class ReviewExportError(RuntimeError):
        pass

    def _missing(*args, **kwargs):
        raise AssertionError("low_light_dataset.review_export is not implemented")

    assign_second_review_requirements = _missing
    export_reviewed_bundle = _missing
    validate_review = _missing


FIRST_REVIEW = "2026-07-17T01:00:00+00:00"
SECOND_REVIEW = "2026-07-17T02:00:00+00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path, format="JPEG", quality=95)


def positive_patch(**changes):
    patch = {
        "status": "positive",
        "anchors": [
            {"y": 120, "x": 240.0, "confidence": 1.0, "source": "human"},
            {"y": 300, "x": 300.0, "confidence": 1.0, "source": "human"},
            {"y": 479, "x": 340.0, "confidence": 1.0, "source": "human"},
        ],
        "first_reviewed_at": FIRST_REVIEW,
    }
    patch.update(changes)
    return patch


def negative_patch(**changes):
    patch = {
        "status": "hard_negative",
        "interference_tags": ["cable_or_rope"],
        "first_reviewed_at": FIRST_REVIEW,
    }
    patch.update(changes)
    return patch


def excluded_patch(**changes):
    patch = {
        "status": "excluded",
        "exclusion_reason": "no_training_value",
        "first_reviewed_at": FIRST_REVIEW,
    }
    patch.update(changes)
    return patch


class ReviewExportFixture(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.candidate_root = self.root / "candidates"
        self.frames_root = self.candidate_root / "frames"
        self.frames_root.mkdir(parents=True)
        self.work_root = self.root / "work"
        self.dataset_root = self.root / "dataset"
        self.make_dataset()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_dataset(self):
        for split, stem, color in (
            ("train", "formal_train", (12, 13, 14)),
            ("test", "formal_test", (22, 23, 24)),
        ):
            save_jpeg(self.dataset_root / split / "pic" / f"{stem}.jpg", color)
            label = np.zeros((480, 640), dtype=np.uint8)
            label[200:400, 310] = 1
            save_label(self.dataset_root / split / "label" / f"{stem}.png", label)

    def prepare(self, stems):
        rows = []
        for index, stem in enumerate(stems):
            path = self.frames_root / f"{stem}.jpg"
            save_jpeg(path, (50 + index * 7, 60 + index * 5, 70 + index * 3))
            rows.append(
                {
                    "stem": stem,
                    "target_timestamp_seconds": str(index),
                    "image_sha256": sha256(path),
                }
            )
        with (self.candidate_root / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        store = ReviewStore(self.work_root, self.candidate_root)
        store.initialize(self.candidate_root / "manifest.csv")
        return store

    @staticmethod
    def error_codes(report):
        return {item["code"] for item in report["errors"]}

    def make_bundle(self, records, root=None):
        bundle = root or (self.root / "bundle")
        (bundle / "pic").mkdir(parents=True)
        (bundle / "label").mkdir()
        for stem, kind in records:
            shutil.copyfile(
                self.frames_root / f"{stem}.jpg", bundle / "pic" / f"{stem}.jpg"
            )
            if kind == "positive":
                label = rasterize_centerline(
                    [
                        AnchorPrediction(120, 240.0, 1.0, "human"),
                        AnchorPrediction(300, 300.0, 1.0, "human"),
                        AnchorPrediction(479, 340.0, 1.0, "human"),
                    ]
                )
            else:
                label = np.zeros((480, 640), dtype=np.uint8)
            save_label(bundle / "label" / f"{stem}.png", label)
        return bundle


class AssignmentTests(ReviewExportFixture):
    def test_assignment_is_deterministic_monotonic_and_uses_audited_updates(self):
        plain = [f"plain_{index:02d}" for index in range(10)]
        stems = ["hard", "modified", "warning", "tagged", "historical", *plain]
        store = self.prepare(stems)
        store.update("hard", negative_patch(), 0)
        store.update("modified", positive_patch(suggestion_modified=True), 0)
        store.update(
            "warning",
            positive_patch(warnings=["temporal_prediction_disagreement"]),
            0,
        )
        store.update(
            "tagged",
            positive_patch(interference_tags=["shadow_or_reflection"]),
            0,
        )
        store.update(
            "historical",
            {
                "status": "needs_second_review",
                "second_review_required": True,
            },
            0,
        )
        store.update("historical", positive_patch(second_review_required=False), 1)
        for stem in plain:
            store.update(stem, positive_patch(), 0)
        store.update("plain_09", {"second_review_required": True}, 1)
        before = {
            item["stem"]: item
            for item in json.loads(
                (self.work_root / "annotation_state.json").read_text(encoding="utf-8")
            )["records"]
        }

        report = assign_second_review_requirements(
            self.candidate_root, self.work_root, seed=71
        )

        plain_pool = ["historical", *plain]
        expected_samples = sorted(
            sorted(
                plain_pool,
                key=lambda stem: hashlib.sha256(f"71:{stem}".encode()).hexdigest(),
            )[: math.ceil(len(plain_pool) * 0.10)]
        )
        self.assertEqual(report["sampled_stems"], expected_samples)
        self.assertEqual(
            set(report["mandatory_stems"]),
            {"hard", "modified", "warning", "tagged", "historical", "plain_09"},
        )
        self.assertIn("plain_09", report["already_required_stems"])
        self.assertEqual(
            set(report["newly_required_stems"]),
            {"hard", "modified", "warning", "tagged", "historical", *expected_samples}
            - {"plain_09"},
        )
        after_document = json.loads(
            (self.work_root / "annotation_state.json").read_text(encoding="utf-8")
        )
        after = {item["stem"]: item for item in after_document["records"]}
        for stem in report["newly_required_stems"]:
            self.assertTrue(after[stem]["second_review_required"])
            for key in (
                "status",
                "anchors",
                "first_reviewed_at",
                "second_reviewed_at",
            ):
                self.assertEqual(after[stem][key], before[stem][key])
        history = [
            json.loads(line)
            for line in (self.work_root / "annotation_history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assignment_events = history[-len(report["newly_required_stems"]) :]
        self.assertTrue(all(event["origin"] == "human" for event in assignment_events))
        self.assertTrue(all(event["actor"].strip() for event in assignment_events))

        state_bytes = (self.work_root / "annotation_state.json").read_bytes()
        second = assign_second_review_requirements(
            self.candidate_root, self.work_root, seed=71
        )
        self.assertEqual(second["newly_required_stems"], [])
        self.assertEqual((self.work_root / "annotation_state.json").read_bytes(), state_bytes)

    def test_assignment_counts_qualifying_history_from_assignment_actor(self):
        store = self.prepare(["historical", "plain"])
        store.update("historical", positive_patch(), 0)
        store.update("plain", positive_patch(), 0)
        store.update(
            "historical",
            {"second_review_required": True},
            1,
            actor="second_review_assignment",
            origin="human",
        )
        store.update(
            "historical",
            {"second_review_required": False},
            2,
            actor="second_review_assignment",
            origin="human",
        )

        report = assign_second_review_requirements(
            self.candidate_root, self.work_root, seed=0
        )

        self.assertIn("historical", report["mandatory_stems"])
        self.assertIn("audit_history", report["reasons"]["historical"])
        self.assertIn("historical", report["newly_required_stems"])
        second = assign_second_review_requirements(
            self.candidate_root, self.work_root, seed=0
        )
        self.assertEqual(second["newly_required_stems"], [])


class ValidationTests(ReviewExportFixture):
    def test_incomplete_expected_coverage_is_reported(self):
        self.prepare(["only"])

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=2
        )

        self.assertFalse(report["ok"])
        self.assertIn("expected_count_mismatch", self.error_codes(report))

    def test_unreviewed_and_pending_second_review_are_both_reported(self):
        store = self.prepare(["pending", "unreviewed"])
        store.update(
            "pending",
            positive_patch(second_review_required=True, second_reviewed_at=None),
            0,
        )

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=2
        )

        self.assertFalse(report["ok"])
        self.assertIn("status_unreviewed", self.error_codes(report))
        self.assertIn("second_review_missing", self.error_codes(report))

    def test_candidate_source_hash_change_is_reported(self):
        store = self.prepare(["changed"])
        store.update("changed", positive_patch(), 0)
        (self.frames_root / "changed.jpg").write_bytes(b"changed after review")

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=1
        )

        self.assertFalse(report["ok"])
        self.assertIn("candidate_hash_mismatch", self.error_codes(report))

    def test_formal_test_set_hash_leakage_is_reported(self):
        store = self.prepare(["leaked"])
        store.update("leaked", positive_patch(), 0)
        shutil.copyfile(
            self.frames_root / "leaked.jpg",
            self.dataset_root / "test" / "pic" / "formal_test.jpg",
        )

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=1
        )

        self.assertFalse(report["ok"])
        self.assertIn("formal_test_hash_collision", self.error_codes(report))

    def test_candidate_formal_filename_collision_is_reported(self):
        store = self.prepare(["formal_train"])
        store.update("formal_train", positive_patch(), 0)

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=1
        )

        self.assertFalse(report["ok"])
        self.assertIn("formal_stem_collision", self.error_codes(report))

    def test_legitimate_dotted_formal_stems_are_safe(self):
        store = self.prepare(["candidate"])
        store.update("candidate", positive_patch(), 0)
        formal_stem = "video1_10.0x_frame_00000"
        for split, color in (("train", (31, 32, 33)), ("test", (41, 42, 43))):
            save_jpeg(
                self.dataset_root / split / "pic" / f"{formal_stem}.jpg", color
            )
            label = np.zeros((480, 640), dtype=np.uint8)
            label[200:400, 310] = 1
            save_label(
                self.dataset_root / split / "label" / f"{formal_stem}.png", label
            )

        report = validate_review(
            self.candidate_root, self.work_root, self.dataset_root, expected_count=1
        )

        self.assertNotIn("formal_stem_unsafe", self.error_codes(report))

    def test_bundle_copy_must_be_byte_identical_to_original(self):
        store = self.prepare(["positive"])
        store.update("positive", positive_patch(), 0)
        bundle = self.make_bundle([("positive", "positive")])
        save_jpeg(bundle / "pic" / "positive.jpg", (1, 2, 3))

        report = validate_review(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            expected_count=1,
            bundle_root=bundle,
        )

        self.assertFalse(report["ok"])
        self.assertIn("bundle_image_hash_mismatch", self.error_codes(report))

    def test_hard_negative_nonzero_bundle_label_is_rejected(self):
        store = self.prepare(["negative"])
        store.update(
            "negative",
            negative_patch(second_review_required=True, second_reviewed_at=SECOND_REVIEW),
            0,
        )
        bundle = self.make_bundle([("negative", "hard_negative")])
        bad = np.zeros((480, 640), dtype=np.uint8)
        bad[300, 300] = 1
        save_label(bundle / "label" / "negative.png", bad)

        report = validate_review(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            expected_count=1,
            bundle_root=bundle,
        )

        self.assertFalse(report["ok"])
        self.assertIn("hard_negative_label_nonzero", self.error_codes(report))

    def test_excluded_pair_leakage_is_rejected(self):
        store = self.prepare(["excluded", "positive"])
        store.update("excluded", excluded_patch(), 0)
        store.update("positive", positive_patch(), 0)
        bundle = self.make_bundle(
            [("positive", "positive"), ("excluded", "hard_negative")]
        )

        report = validate_review(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            expected_count=2,
            bundle_root=bundle,
        )

        self.assertFalse(report["ok"])
        self.assertIn("excluded_pair_leakage", self.error_codes(report))

    def test_mismatched_paired_stems_are_rejected(self):
        store = self.prepare(["positive"])
        store.update("positive", positive_patch(), 0)
        bundle = self.make_bundle([("positive", "positive")])
        (bundle / "label" / "positive.png").replace(bundle / "label" / "other.png")

        report = validate_review(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            expected_count=1,
            bundle_root=bundle,
        )

        self.assertFalse(report["ok"])
        self.assertIn("bundle_pair_mismatch", self.error_codes(report))


class ExportTests(ReviewExportFixture):
    def prepare_fully_reviewed(self):
        store = self.prepare(["negative", "positive", "excluded"])
        store.update("negative", negative_patch(), 0)
        store.update("positive", positive_patch(), 0)
        store.update("excluded", excluded_patch(), 0)
        assign_second_review_requirements(self.candidate_root, self.work_root, seed=17)
        for stem in ("negative", "positive"):
            record = store.get(stem)
            if record.second_review_required:
                store.update(
                    stem,
                    {"second_reviewed_at": SECOND_REVIEW},
                    record.revision,
                )
        return store

    def test_existing_output_and_stale_staging_are_refused(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        output.mkdir()
        with self.assertRaisesRegex(ReviewExportError, "output_exists"):
            export_reviewed_bundle(
                self.candidate_root,
                self.work_root,
                self.dataset_root,
                output,
                expected_count=3,
                seed=17,
            )
        output.rmdir()
        output.with_name(output.name + ".staging").mkdir()
        with self.assertRaisesRegex(ReviewExportError, "stale_staging"):
            export_reviewed_bundle(
                self.candidate_root,
                self.work_root,
                self.dataset_root,
                output,
                expected_count=3,
                seed=17,
            )

    def test_export_requires_explicit_assignment_without_mutating_review_state(self):
        store = self.prepare(["positive"])
        store.update("positive", positive_patch(), 0)
        before_state = (self.work_root / "annotation_state.json").read_bytes()
        before_history = (self.work_root / "annotation_history.jsonl").read_bytes()

        with self.assertRaisesRegex(
            ReviewExportError, "second_review_assignment_required"
        ):
            export_reviewed_bundle(
                self.candidate_root,
                self.work_root,
                self.dataset_root,
                self.root / "reviewed",
                expected_count=1,
                seed=17,
            )

        self.assertEqual(
            (self.work_root / "annotation_state.json").read_bytes(), before_state
        )
        self.assertEqual(
            (self.work_root / "annotation_history.jsonl").read_bytes(), before_history
        )
        self.assertFalse((self.root / "reviewed.staging").exists())

    def test_output_reparse_ancestor_is_rejected(self):
        self.prepare_fully_reviewed()
        publish_parent = self.root / "publish"
        publish_parent.mkdir()
        output = publish_parent / "reviewed"
        original = review_export_module._is_link_like

        def link_like(path):
            return Path(path) == publish_parent or original(Path(path))

        with mock.patch.object(review_export_module, "_is_link_like", side_effect=link_like):
            with self.assertRaisesRegex(ReviewExportError, "unsafe_output_location"):
                export_reviewed_bundle(
                    self.candidate_root,
                    self.work_root,
                    self.dataset_root,
                    output,
                    expected_count=3,
                    seed=17,
                )
        self.assertFalse(output.exists())

    def test_output_ancestry_is_rechecked_immediately_before_publish(self):
        self.prepare_fully_reviewed()
        publish_parent = self.root / "publish"
        publish_parent.mkdir()
        output = publish_parent / "reviewed"
        original = review_export_module._is_link_like
        parent_checks = 0

        def link_like(path):
            nonlocal parent_checks
            path = Path(path)
            if path == publish_parent:
                parent_checks += 1
                return parent_checks >= 2
            return original(path)

        with mock.patch.object(review_export_module, "_is_link_like", side_effect=link_like):
            with self.assertRaisesRegex(ReviewExportError, "unsafe_output_location"):
                export_reviewed_bundle(
                    self.candidate_root,
                    self.work_root,
                    self.dataset_root,
                    output,
                    expected_count=3,
                    seed=17,
                )
        self.assertGreaterEqual(parent_checks, 2)
        self.assertFalse(output.exists())
        self.assertTrue(output.with_name(output.name + ".staging").exists())

    def test_overlay_is_generated_from_verified_staging_copy(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        with mock.patch.object(
            review_export_module, "load_bgr", wraps=load_bgr
        ) as loader:
            export_reviewed_bundle(
                self.candidate_root,
                self.work_root,
                self.dataset_root,
                output,
                expected_count=3,
                seed=17,
            )

        loaded_paths = [Path(call.args[0]) for call in loader.call_args_list]
        self.assertTrue(loaded_paths)
        self.assertTrue(
            all(path.parent.name == "pic" and path.parent.parent.name == "reviewed.staging"
                for path in loaded_paths)
        )

    def test_staging_validation_decodes_overlay_and_contact_sheet_jpegs(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        export_reviewed_bundle(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            output,
            expected_count=3,
            seed=17,
        )
        for relative in (
            Path("overlay") / "positive.jpg",
            Path("contact_sheets") / "overlay_contact_001.jpg",
        ):
            with self.subTest(relative=relative):
                path = output / relative
                original = path.read_bytes()
                path.write_bytes(b"not a jpeg")

                report = validate_review(
                    self.candidate_root,
                    self.work_root,
                    self.dataset_root,
                    expected_count=3,
                    bundle_root=output,
                    seed=17,
                )

                self.assertFalse(report["ok"])
                self.assertIn("bundle_jpeg_invalid", self.error_codes(report))
                path.write_bytes(original)

    def test_staging_validation_checks_report_schema_counts_stems_and_hashes(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        export_reviewed_bundle(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            output,
            expected_count=3,
            seed=17,
        )
        path = output / "review_report.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schema_version"] = 99
        document["exported_count"] = 999
        document["exported_stems"] = ["wrong"]
        document["file_hashes"]["positive"]["image_sha256"] = "0" * 64
        path.write_text(json.dumps(document), encoding="utf-8")

        report = validate_review(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            expected_count=3,
            bundle_root=output,
            seed=17,
        )

        self.assertFalse(report["ok"])
        self.assertIn("bundle_report_mismatch", self.error_codes(report))

    def test_staging_validation_checks_assignment_snapshots_and_all_validation_fields(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        export_reviewed_bundle(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            output,
            expected_count=3,
            seed=17,
        )
        mutations = (
            ("review_report.json", lambda document: document.__setitem__("second_review_assignment", {})),
            (
                "review_report.json",
                lambda document: (
                    document.__setitem__("formal_dataset_before", {}),
                    document.__setitem__("formal_dataset_after", {}),
                ),
            ),
            ("validation_report.json", lambda document: document.pop("status_counts")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name, mutation=str(mutate)):
                path = output / name
                original = path.read_bytes()
                document = json.loads(original.decode("utf-8"))
                mutate(document)
                path.write_text(json.dumps(document), encoding="utf-8")

                report = validate_review(
                    self.candidate_root,
                    self.work_root,
                    self.dataset_root,
                    expected_count=3,
                    bundle_root=output,
                    seed=17,
                )

                self.assertFalse(report["ok"])
                self.assertIn("bundle_report_mismatch", self.error_codes(report))
                path.write_bytes(original)

    def test_report_hash_read_failure_is_a_structured_validation_error(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        export_reviewed_bundle(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            output,
            expected_count=3,
            seed=17,
        )
        unreadable = output / "label" / "positive.png"
        original_sha256 = review_export_module._sha256_file

        def fail_one(path):
            if Path(path) == unreadable:
                raise PermissionError("simulated read race")
            return original_sha256(Path(path))

        with mock.patch.object(
            review_export_module, "_sha256_file", side_effect=fail_one
        ):
            report = validate_review(
                self.candidate_root,
                self.work_root,
                self.dataset_root,
                expected_count=3,
                bundle_root=output,
                seed=17,
            )

        self.assertFalse(report["ok"])
        self.assertIn("bundle_report_unreadable", self.error_codes(report))

    def test_successful_paired_export_has_reports_and_preserves_formal_dataset(self):
        self.prepare_fully_reviewed()
        output = self.root / "reviewed"
        formal_before = snapshot_dataset(self.dataset_root)

        result = export_reviewed_bundle(
            self.candidate_root,
            self.work_root,
            self.dataset_root,
            output,
            expected_count=3,
            seed=17,
        )

        self.assertEqual(result, output)
        self.assertEqual(snapshot_dataset(self.dataset_root), formal_before)
        self.assertEqual(
            {path.stem for path in (output / "pic").glob("*.jpg")},
            {"negative", "positive"},
        )
        self.assertEqual(
            {path.stem for path in (output / "label").glob("*.png")},
            {"negative", "positive"},
        )
        self.assertFalse((output / "pic" / "excluded.jpg").exists())
        self.assertFalse((output / "label" / "excluded.png").exists())
        self.assertEqual(
            (output / "pic" / "positive.jpg").read_bytes(),
            (self.frames_root / "positive.jpg").read_bytes(),
        )
        with Image.open(output / "label" / "negative.png") as image:
            self.assertEqual(int(np.asarray(image).sum()), 0)
        for name in ("annotation.json", "review_report.json", "validation_report.json"):
            document = json.loads((output / name).read_text(encoding="utf-8"))
            self.assertIsInstance(document, dict)
        sheets = sorted((output / "contact_sheets").glob("overlay_contact_*.jpg"))
        self.assertEqual(len(sheets), math.ceil(2 / 20))
        validation = json.loads(
            (output / "validation_report.json").read_text(encoding="utf-8")
        )
        review = json.loads((output / "review_report.json").read_text(encoding="utf-8"))
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual(review["exported_count"], 2)
        self.assertEqual(review["excluded_count"], 1)
        self.assertEqual(review["formal_dataset_before"], review["formal_dataset_after"])


if __name__ == "__main__":
    unittest.main()
