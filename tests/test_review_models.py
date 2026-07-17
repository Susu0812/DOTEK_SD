import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace

from low_light_dataset.annotation import AnchorPrediction
from low_light_dataset.review_models import (
    ReviewRecord,
    ReviewStatus,
    record_from_dict,
    record_to_dict,
    validate_record,
)


VALID_HASH = "a" * 64


def make_record(**changes):
    values = {
        "stem": "lowlight_camera_full_rgb_t000000",
        "image_sha256": VALID_HASH,
        "revision": 1,
        "status": ReviewStatus.POSITIVE,
        "anchors": (
            AnchorPrediction(220, 280.0, 0.91, "model"),
            AnchorPrediction(320, 300.5, 0.83, "human"),
            AnchorPrediction(479, 330.0, 0.95, "human"),
        ),
        "interference_tags": (),
        "exclusion_reason": None,
        "warnings": (),
        "suggestion_modified": True,
        "first_reviewed_at": "2026-07-16T10:00:00+08:00",
        "second_review_required": False,
        "second_reviewed_at": None,
        "notes": "target hose follows the centerline",
    }
    values.update(changes)
    return ReviewRecord(**values)


class ReviewRecordTests(unittest.TestCase):
    def test_record_is_immutable(self):
        record = make_record()

        with self.assertRaises(FrozenInstanceError):
            record.revision = 2

    def test_valid_status_records_have_no_errors(self):
        records = [
            make_record(),
            make_record(
                status=ReviewStatus.HARD_NEGATIVE,
                anchors=(),
                interference_tags=("cable_or_rope",),
            ),
            make_record(
                status=ReviewStatus.EXCLUDED,
                anchors=(),
                exclusion_reason="severe_motion_blur",
            ),
            make_record(
                status=ReviewStatus.UNREVIEWED,
                anchors=(),
                first_reviewed_at=None,
                suggestion_modified=False,
            ),
            make_record(
                status=ReviewStatus.NEEDS_SECOND_REVIEW,
                second_review_required=True,
                second_reviewed_at=None,
            ),
        ]

        for record in records:
            with self.subTest(status=record.status):
                self.assertEqual(validate_record(record), [])

    def test_positive_requires_three_anchors(self):
        record = make_record(anchors=make_record().anchors[:2])

        self.assertIn("positive_requires_three_anchors", validate_record(record))

    def test_hard_negative_rejects_anchors_and_requires_allowed_tag(self):
        with_anchors = make_record(
            status=ReviewStatus.HARD_NEGATIVE,
            interference_tags=("shadow_or_reflection",),
        )
        without_tag = replace(with_anchors, anchors=(), interference_tags=())

        self.assertIn("hard_negative_has_anchors", validate_record(with_anchors))
        self.assertIn(
            "hard_negative_requires_interference_tag",
            validate_record(without_tag),
        )

    def test_exclusion_reason_rules_follow_status(self):
        missing = make_record(
            status=ReviewStatus.EXCLUDED,
            anchors=(),
            exclusion_reason=None,
        )
        on_positive = make_record(exclusion_reason="near_duplicate")

        self.assertIn("excluded_requires_exclusion_reason", validate_record(missing))
        self.assertIn("non_excluded_has_exclusion_reason", validate_record(on_positive))


class FieldValidationTests(unittest.TestCase):
    def test_rejects_empty_stem_negative_revision_and_bad_hashes(self):
        invalid_hashes = ("a" * 63, "A" * 64, "g" * 64)
        self.assertIn("stem_empty", validate_record(make_record(stem="")))
        self.assertIn(
            "revision_negative", validate_record(make_record(revision=-1))
        )
        for value in invalid_hashes:
            with self.subTest(value=value):
                self.assertIn(
                    "image_sha256_invalid",
                    validate_record(make_record(image_sha256=value)),
                )

    def test_rejects_unknown_allowed_values(self):
        unknown_tag = make_record(interference_tags=("wet_floor",))
        unknown_reason = make_record(
            status=ReviewStatus.EXCLUDED,
            anchors=(),
            exclusion_reason="operator_unsure",
        )

        self.assertIn("unknown_interference_tag", validate_record(unknown_tag))
        self.assertIn("unknown_exclusion_reason", validate_record(unknown_reason))

    def test_rejects_descending_and_duplicate_anchor_rows(self):
        descending = make_record(
            anchors=(
                AnchorPrediction(300, 10.0, 0.8, "human"),
                AnchorPrediction(200, 20.0, 0.8, "human"),
                AnchorPrediction(400, 30.0, 0.8, "human"),
            )
        )
        duplicate = make_record(
            anchors=(
                AnchorPrediction(200, 10.0, 0.8, "human"),
                AnchorPrediction(200, 20.0, 0.8, "human"),
                AnchorPrediction(400, 30.0, 0.8, "human"),
            )
        )
        non_adjacent_duplicate = make_record(
            anchors=(
                AnchorPrediction(100, 10.0, 0.8, "human"),
                AnchorPrediction(200, 20.0, 0.8, "human"),
                AnchorPrediction(100, 30.0, 0.8, "human"),
            )
        )

        self.assertIn("anchor_y_not_increasing", validate_record(descending))
        self.assertIn("duplicate_anchor_y", validate_record(duplicate))
        self.assertIn(
            "duplicate_anchor_y", validate_record(non_adjacent_duplicate)
        )

    def test_rejects_non_finite_and_out_of_bounds_coordinates(self):
        non_finite = make_record(
            anchors=(
                AnchorPrediction(100, math.nan, 0.8, "human"),
                AnchorPrediction(200, 20.0, 0.8, "human"),
                AnchorPrediction(300, 30.0, 0.8, "human"),
            )
        )
        bad_coordinates = (
            AnchorPrediction(100, -0.1, 0.8, "human"),
            AnchorPrediction(200, 640.0, 0.8, "human"),
            AnchorPrediction(480, 30.0, 0.8, "human"),
        )

        self.assertIn("anchor_coordinate_non_finite", validate_record(non_finite))
        self.assertIn(
            "anchor_out_of_bounds",
            validate_record(make_record(anchors=bad_coordinates)),
        )


class TimestampValidationTests(unittest.TestCase):
    def test_unreviewed_rejects_first_or_second_review_timestamps(self):
        first = make_record(
            status=ReviewStatus.UNREVIEWED,
            anchors=(),
            first_reviewed_at="2026-07-16T10:00:00+08:00",
        )
        second = replace(
            first,
            first_reviewed_at=None,
            second_review_required=True,
            second_reviewed_at="2026-07-16T11:00:00+08:00",
        )

        self.assertIn("unreviewed_has_review_timestamp", validate_record(first))
        self.assertIn("unreviewed_has_review_timestamp", validate_record(second))

    def test_final_statuses_require_first_review_timestamp(self):
        for status, changes in (
            (ReviewStatus.POSITIVE, {}),
            (
                ReviewStatus.HARD_NEGATIVE,
                {"anchors": (), "interference_tags": ("elongated_object",)},
            ),
            (
                ReviewStatus.EXCLUDED,
                {"anchors": (), "exclusion_reason": "no_training_value"},
            ),
        ):
            record = make_record(
                status=status,
                first_reviewed_at=None,
                **changes,
            )
            with self.subTest(status=status):
                self.assertIn(
                    "final_status_requires_first_review",
                    validate_record(record),
                )

    def test_second_review_timestamp_requires_second_review_flag(self):
        record = make_record(second_reviewed_at="2026-07-16T11:00:00+08:00")

        self.assertIn(
            "second_reviewed_without_requirement", validate_record(record)
        )

    def test_pending_required_second_review_is_valid_at_record_level(self):
        record = make_record(
            second_review_required=True,
            second_reviewed_at=None,
        )

        self.assertEqual(validate_record(record), [])


class SerializationTests(unittest.TestCase):
    def test_json_round_trip_preserves_anchor_confidence_and_source(self):
        record = make_record(
            interference_tags=("floor_seam_or_edge", "non_target_hose"),
            warnings=("prediction_disagreement_y_320",),
        )

        encoded = record_to_dict(record)
        decoded = record_from_dict(json.loads(json.dumps(encoded)))

        self.assertEqual(encoded["status"], "positive")
        self.assertIsInstance(encoded["anchors"], list)
        self.assertEqual(encoded["anchors"][0]["confidence"], 0.91)
        self.assertEqual(encoded["anchors"][0]["source"], "model")
        self.assertEqual(decoded, record)

    def test_from_dict_converts_lists_and_anchor_dicts(self):
        data = record_to_dict(make_record())

        record = record_from_dict(data)

        self.assertIsInstance(record.status, ReviewStatus)
        self.assertIsInstance(record.anchors, tuple)
        self.assertIsInstance(record.anchors[0], AnchorPrediction)
        self.assertIsInstance(record.interference_tags, tuple)
        self.assertIsInstance(record.warnings, tuple)

    def test_from_dict_rejects_missing_required_field(self):
        data = record_to_dict(make_record())
        del data["revision"]

        with self.assertRaises((KeyError, TypeError)):
            record_from_dict(data)

    def test_from_dict_rejects_invalid_status_value(self):
        data = record_to_dict(make_record())
        data["status"] = "approved"

        with self.assertRaises(ValueError):
            record_from_dict(data)


if __name__ == "__main__":
    unittest.main()
