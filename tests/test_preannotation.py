import copy
import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from low_light_dataset.annotation import AnchorPrediction
from low_light_dataset.image_ops import (
    EnhancedFrame,
    EnhancementParams,
    FrameMetrics,
)
from low_light_dataset import preannotation
from low_light_dataset.preannotation import add_temporal_warnings, build_preannotations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jpeg(path: Path, value: int, size: tuple[int, int] = (640, 480)) -> None:
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    path.write_bytes(encoded.tobytes())


def _metrics(value: float) -> FrameMetrics:
    return FrameMetrics(
        mean=value,
        median=value,
        p05=value,
        p95=value,
        dark_fraction=0.25,
        highlight_fraction=0.0,
        laplacian_variance=1.5,
    )


def _fake_enhance(frame: np.ndarray) -> EnhancedFrame:
    enhanced = np.clip(frame.astype(np.int16) + 7, 0, 255).astype(np.uint8)
    return EnhancedFrame(
        image=enhanced,
        before=_metrics(float(frame.mean())),
        after=_metrics(float(enhanced.mean())),
        params=EnhancementParams(
            gamma=0.8,
            retinex_weight=0.4,
            clahe_clip_limit=2.0,
            denoise_sigma_color=30.0,
            sharpen_amount=0.2,
        ),
    )


class CandidateFixture:
    def __init__(
        self,
        base: Path,
        rows: list[tuple[str, float, int]] | None = None,
    ) -> None:
        self.root = base / "candidates"
        self.frames = self.root / "frames"
        self.frames.mkdir(parents=True)
        self.checkpoint = base / "model.pth"
        self.checkpoint.write_bytes(b"fake-checkpoint-for-preannotation-tests")
        self.rows = rows or [("frame_000010", 10.0, 30), ("frame_000002", 2.0, 20)]
        for stem, _, value in self.rows:
            _write_jpeg(self.frames / f"{stem}.jpg", value)
        self.write_manifest()

    def manifest_rows(self) -> list[dict[str, str]]:
        return [
            {
                "stem": stem,
                "target_timestamp_seconds": str(timestamp),
                "image_sha256": _sha256(self.frames / f"{stem}.jpg"),
                "source": "test",
            }
            for stem, timestamp, _ in self.rows
        ]

    def write_manifest(
        self,
        rows: list[dict[str, str]] | None = None,
        fieldnames: list[str] | None = None,
    ) -> None:
        actual_rows = rows if rows is not None else self.manifest_rows()
        actual_fields = fieldnames or [
            "stem",
            "target_timestamp_seconds",
            "image_sha256",
            "source",
        ]
        with (self.root / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=actual_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(actual_rows)

    def tree_snapshot(self) -> dict[str, tuple[str, bytes]]:
        return {
            path.relative_to(self.root).as_posix(): (
                "file" if path.is_file() else "directory",
                path.read_bytes() if path.is_file() else b"",
            )
            for path in sorted(self.root.rglob("*"))
        }


class FakeAnnotator:
    devices: list[str] = []
    frames: list[np.ndarray] = []

    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        self.checkpoint_path = checkpoint_path
        type(self).devices.append(device)

    def predict(self, frame_bgr: np.ndarray) -> list[AnchorPrediction]:
        type(self).frames.append(frame_bgr.copy())
        offset = float(frame_bgr.mean())
        return [
            AnchorPrediction(220, 100.0 + offset, 0.8, "model"),
            AnchorPrediction(320, 110.0 + offset, 0.8, "model"),
            AnchorPrediction(420, 120.0 + offset, 0.8, "model"),
        ]


def _fake_color(frame_bgr: np.ndarray) -> list[AnchorPrediction]:
    offset = float(frame_bgr.mean())
    return [
        AnchorPrediction(220, 102.0 + offset, 0.7, "color"),
        AnchorPrediction(320, 112.0 + offset, 0.7, "color"),
        AnchorPrediction(420, 122.0 + offset, 0.7, "color"),
    ]


class PreannotationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAnnotator.devices = []
        FakeAnnotator.frames = []

    def _patch_lightweight_pipeline(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(mock.patch.object(preannotation, "HoseAnnotator", FakeAnnotator))
        stack.enter_context(mock.patch.object(preannotation, "enhance_low_light", _fake_enhance))
        stack.enter_context(mock.patch.object(preannotation, "extract_color_anchors", _fake_color))
        return stack

    def test_uses_original_and_enhanced_sources_with_required_fusion_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root, [("only", 3.5, 25)])
            output = root / "preannotations.json"
            extraction_frames: list[np.ndarray] = []
            fuse_calls: list[tuple[list[AnchorPrediction], list[AnchorPrediction], float]] = []
            regularize_calls: list[list[AnchorPrediction]] = []

            class SourceWarningAnnotator(FakeAnnotator):
                def predict(self, frame_bgr: np.ndarray):
                    return super().predict(frame_bgr), [
                        "model_source_warning",
                        "model_source_warning",
                    ]

            def extract(frame: np.ndarray):
                extraction_frames.append(frame.copy())
                source = "color_original" if len(extraction_frames) == 1 else "color_enhanced"
                return [AnchorPrediction(220, 200.0, 0.75, source)], [
                    "color_source_warning",
                    "color_source_warning",
                ]

            def fuse(
                original: list[AnchorPrediction],
                enhanced: list[AnchorPrediction],
                max_disagreement_px: float = 48.0,
            ) -> tuple[list[AnchorPrediction], list[str]]:
                fuse_calls.append((original, enhanced, max_disagreement_px))
                return [
                    AnchorPrediction(
                        220,
                        100.0 + max_disagreement_px,
                        0.9,
                        f"fused_{int(max_disagreement_px)}",
                    )
                ], [f"fusion_warning_{int(max_disagreement_px)}"]

            def regularize(
                predictions: list[AnchorPrediction],
            ) -> tuple[list[AnchorPrediction], list[str]]:
                regularize_calls.append(predictions)
                return predictions, ["regularization_warning"]

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(preannotation, "HoseAnnotator", SourceWarningAnnotator)
                )
                stack.enter_context(mock.patch.object(preannotation, "enhance_low_light", _fake_enhance))
                stack.enter_context(mock.patch.object(preannotation, "extract_color_anchors", extract))
                stack.enter_context(mock.patch.object(preannotation, "fuse_predictions", fuse))
                stack.enter_context(mock.patch.object(preannotation, "regularize_anchors", regularize))
                result = build_preannotations(
                    fixture.root, fixture.checkpoint, output, device="cpu"
                )

            self.assertEqual(result, output)
            self.assertEqual([call[2] for call in fuse_calls], [48.0, 64.0, 96.0])
            self.assertEqual(len(FakeAnnotator.frames), 2)
            self.assertEqual(len(extraction_frames), 2)
            self.assertFalse(np.array_equal(FakeAnnotator.frames[0], FakeAnnotator.frames[1]))
            self.assertTrue(np.array_equal(FakeAnnotator.frames[0], extraction_frames[0]))
            self.assertTrue(np.array_equal(FakeAnnotator.frames[1], extraction_frames[1]))
            self.assertEqual(fuse_calls[0][0][0].source, "model")
            self.assertEqual(fuse_calls[0][1][0].source, "model")
            self.assertEqual(fuse_calls[1][0][0].source, "color_original")
            self.assertEqual(fuse_calls[1][1][0].source, "color_enhanced")
            self.assertEqual(fuse_calls[2][0][0].source, "fused_48")
            self.assertEqual(fuse_calls[2][1][0].source, "fused_64")
            self.assertEqual(regularize_calls[0][0].source, "fused_96")

            document = json.loads(output.read_text(encoding="utf-8"))
            record = document["records"][0]
            self.assertEqual(
                record["warnings"],
                [
                    "color_source_warning",
                    "fusion_warning_48",
                    "fusion_warning_64",
                    "fusion_warning_96",
                    "insufficient_anchor_count",
                    "model_source_warning",
                    "regularization_warning",
                ],
            )
            self.assertEqual(record["source_metrics"]["model"]["original_count"], 3)
            self.assertEqual(record["source_metrics"]["model"]["enhanced_count"], 3)
            self.assertEqual(record["source_metrics"]["model"]["fused_count"], 1)
            self.assertEqual(record["source_metrics"]["color"]["original_count"], 1)
            self.assertEqual(record["source_metrics"]["color"]["enhanced_count"], 1)
            self.assertEqual(record["source_metrics"]["color"]["fused_count"], 1)
            self.assertEqual(record["source_metrics"]["combined_fused_count"], 1)
            self.assertEqual(record["source_metrics"]["final_count"], 1)
            self.assertIn("before", record["source_metrics"]["enhancement"])
            self.assertIn("after", record["source_metrics"]["enhancement"])
            self.assertIn("params", record["source_metrics"]["enhancement"])

    def test_cpu_smoke_is_deterministic_covers_exact_stems_and_keeps_candidates_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root)
            before = fixture.tree_snapshot()
            first = root / "first.json"
            second = root / "second.json"

            with self._patch_lightweight_pipeline():
                build_preannotations(fixture.root, fixture.checkpoint, first, device="cpu")
                build_preannotations(fixture.root, fixture.checkpoint, second, device="cpu")

            self.assertEqual(first.read_bytes(), second.read_bytes())
            document = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["checkpoint_sha256"], _sha256(fixture.checkpoint))
            self.assertEqual(
                [record["stem"] for record in document["records"]],
                ["frame_000002", "frame_000010"],
            )
            self.assertEqual(
                {record["stem"] for record in document["records"]},
                {stem for stem, _, _ in fixture.rows},
            )
            self.assertTrue(
                all(isinstance(record["target_timestamp_seconds"], (int, float))
                    for record in document["records"])
            )
            forbidden = {"status", "first_reviewed_at", "second_reviewed_at", "revision"}
            for record in document["records"]:
                self.assertTrue(forbidden.isdisjoint(record))
            self.assertEqual(FakeAnnotator.devices, ["cpu", "cpu"])
            self.assertEqual(fixture.tree_snapshot(), before)
            self.assertFalse(list(fixture.root.rglob("*enhanced*.jpg")))

    def test_validation_failures_do_not_publish_output(self):
        cases = (
            "hash_mismatch",
            "duplicate_stem",
            "malformed_manifest",
            "missing_jpeg",
            "unreadable_jpeg",
            "wrong_size_jpeg",
            "missing_checkpoint",
            "output_inside_candidate",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = CandidateFixture(root)
                rows = fixture.manifest_rows()
                output = root / "result.json"
                checkpoint = fixture.checkpoint
                expected_exception: type[BaseException] = ValueError

                if case == "hash_mismatch":
                    rows[0]["image_sha256"] = "0" * 64
                    fixture.write_manifest(rows)
                elif case == "duplicate_stem":
                    fixture.write_manifest(rows + [dict(rows[0])])
                elif case == "malformed_manifest":
                    fixture.write_manifest(rows, ["stem", "target_timestamp_seconds"])
                elif case == "missing_jpeg":
                    (fixture.frames / f"{rows[0]['stem']}.jpg").unlink()
                    expected_exception = FileNotFoundError
                elif case == "unreadable_jpeg":
                    image_path = fixture.frames / f"{rows[0]['stem']}.jpg"
                    image_path.write_bytes(b"not-a-jpeg")
                    rows[0]["image_sha256"] = _sha256(image_path)
                    fixture.write_manifest(rows)
                elif case == "wrong_size_jpeg":
                    image_path = fixture.frames / f"{rows[0]['stem']}.jpg"
                    _write_jpeg(image_path, 20, size=(320, 240))
                    rows[0]["image_sha256"] = _sha256(image_path)
                    fixture.write_manifest(rows)
                elif case == "missing_checkpoint":
                    checkpoint = root / "missing.pth"
                    expected_exception = FileNotFoundError
                elif case == "output_inside_candidate":
                    output = fixture.root / "suggestions.json"

                with self._patch_lightweight_pipeline(), self.assertRaises(expected_exception):
                    build_preannotations(fixture.root, checkpoint, output, device="cpu")
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_prediction_failure_does_not_publish_partial_json(self):
        class FailingAnnotator(FakeAnnotator):
            def predict(self, frame_bgr: np.ndarray) -> list[AnchorPrediction]:
                if len(type(self).frames) >= 2:
                    raise RuntimeError("synthetic prediction failure")
                return super().predict(frame_bgr)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root)
            output = root / "result.json"
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(preannotation, "HoseAnnotator", FailingAnnotator))
                stack.enter_context(mock.patch.object(preannotation, "enhance_low_light", _fake_enhance))
                stack.enter_context(mock.patch.object(preannotation, "extract_color_anchors", _fake_color))
                with self.assertRaisesRegex(RuntimeError, "synthetic prediction failure"):
                    build_preannotations(fixture.root, fixture.checkpoint, output, device="cpu")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_resolved_output_alias_cannot_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root)
            alias_parent = root / "alias_parent"
            alias_parent.mkdir()
            aliases = [
                fixture.checkpoint,
                alias_parent / ".." / fixture.checkpoint.name,
            ]
            symlink = root / "checkpoint-link.pth"
            try:
                symlink.symlink_to(fixture.checkpoint)
            except OSError:
                pass
            else:
                aliases.append(symlink)

            for output in aliases:
                with self.subTest(output=str(output)):
                    checkpoint_before = fixture.checkpoint.read_bytes()
                    with self._patch_lightweight_pipeline(), self.assertRaises(ValueError):
                        build_preannotations(
                            fixture.root,
                            fixture.checkpoint,
                            output,
                            device="cpu",
                        )
                    self.assertEqual(fixture.checkpoint.read_bytes(), checkpoint_before)

    def test_atomic_write_does_not_touch_predictable_preexisting_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root)
            output = root / "result.json"
            unrelated = output.with_name(output.name + ".tmp")
            unrelated.write_bytes(b"unrelated-temp-sentinel")

            with self._patch_lightweight_pipeline():
                build_preannotations(fixture.root, fixture.checkpoint, output, device="cpu")

            self.assertTrue(output.is_file())
            self.assertEqual(unrelated.read_bytes(), b"unrelated-temp-sentinel")

    def test_replace_failure_keeps_existing_output_and_unrelated_temp_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateFixture(root)
            output = root / "result.json"
            output.write_bytes(b"previous-complete-output")
            unrelated = output.with_name(output.name + ".tmp")
            unrelated.write_bytes(b"unrelated-temp-sentinel")

            with self._patch_lightweight_pipeline(), mock.patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=OSError("synthetic replace failure"),
            ), self.assertRaisesRegex(OSError, "synthetic replace failure"):
                build_preannotations(fixture.root, fixture.checkpoint, output, device="cpu")

            self.assertEqual(output.read_bytes(), b"previous-complete-output")
            self.assertEqual(unrelated.read_bytes(), b"unrelated-temp-sentinel")
            self.assertFalse(list(root.glob(".result.json.*.tmp")))


def _record(stem: str, timestamp: float, xs: list[float], rows: list[int] | None = None) -> dict:
    ys = rows or [200, 300, 400]
    return {
        "stem": stem,
        "target_timestamp_seconds": timestamp,
        "anchors": [
            {"y": y, "x": x, "confidence": 0.9, "source": "model"}
            for y, x in zip(ys, xs)
        ],
        "warnings": [],
    }


class TemporalWarningTests(unittest.TestCase):
    def test_large_median_residual_warns_without_mutating_anchor_geometry(self):
        records = [
            _record("later", 20.0, [120.0, 220.0, 320.0]),
            _record("current", 10.0, [400.0, 500.0, 600.0]),
            _record("earlier", 0.0, [100.0, 200.0, 300.0]),
        ]
        original = copy.deepcopy(records)

        result = add_temporal_warnings(records, threshold_px=96.0)

        self.assertEqual(records, original)
        self.assertEqual([item["stem"] for item in result], ["earlier", "current", "later"])
        current = next(item for item in result if item["stem"] == "current")
        self.assertIn("temporal_disagreement", current["warnings"])
        self.assertEqual(
            [anchor["x"] for anchor in current["anchors"]],
            [400.0, 500.0, 600.0],
        )
        self.assertIsNot(result[0], records[0])

    def test_stable_sequence_and_insufficient_shared_rows_do_not_false_warn(self):
        stable = [
            _record("a", 0.0, [100.0, 200.0, 300.0]),
            _record("b", 10.0, [110.0, 210.0, 310.0]),
            _record("c", 20.0, [120.0, 220.0, 320.0]),
        ]
        sparse = [
            _record("a", 0.0, [100.0], [200]),
            _record("b", 10.0, [500.0], [200]),
            _record("c", 20.0, [120.0], [200]),
        ]

        stable_result = add_temporal_warnings(stable)
        sparse_result = add_temporal_warnings(sparse)

        self.assertNotIn("temporal_disagreement", stable_result[1]["warnings"])
        self.assertNotIn("temporal_disagreement", sparse_result[1]["warnings"])

    def test_fewer_than_three_anchors_adds_unique_insufficient_warning(self):
        records = [_record("one", 1.0, [100.0, 110.0], [200, 300])]
        records[0]["warnings"] = ["insufficient_anchor_count", "existing"]

        result = add_temporal_warnings(records)

        self.assertEqual(
            result[0]["warnings"], ["existing", "insufficient_anchor_count"]
        )

    def test_residual_equal_to_threshold_does_not_warn(self):
        records = [
            _record("earlier", 0.0, [100.0, 200.0, 300.0]),
            _record("current", 10.0, [206.0, 306.0, 406.0]),
            _record("later", 20.0, [120.0, 220.0, 320.0]),
        ]

        result = add_temporal_warnings(records, threshold_px=96.0)

        self.assertNotIn("temporal_disagreement", result[1]["warnings"])

    def test_equal_timestamp_record_is_not_used_as_temporal_neighbor(self):
        records = [
            _record("earlier", 0.0, [100.0, 200.0, 300.0]),
            _record("current", 10.0, [110.0, 210.0, 310.0]),
            _record("same_time_outlier", 10.0, [500.0, 500.0, 500.0]),
            _record("later", 20.0, [120.0, 220.0, 320.0]),
        ]

        result = add_temporal_warnings(records)

        current = next(record for record in result if record["stem"] == "current")
        self.assertNotIn("temporal_disagreement", current["warnings"])


if __name__ == "__main__":
    unittest.main()
