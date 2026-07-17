"""Immutable review records and record-level validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from low_light_dataset.annotation import AnchorPrediction


INTERFERENCE_TAGS = frozenset(
    {
        "floor_seam_or_edge",
        "shadow_or_reflection",
        "cable_or_rope",
        "non_target_hose",
        "elongated_object",
        "motion_or_low_light_artifact",
    }
)

EXCLUSION_REASONS = frozenset(
    {
        "near_duplicate",
        "severe_motion_blur",
        "severe_exposure_failure",
        "ambiguous_target_identity",
        "long_occlusion_unknown_path",
        "no_training_value",
    }
)


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    EXCLUDED = "excluded"
    NEEDS_SECOND_REVIEW = "needs_second_review"


@dataclass(frozen=True)
class ReviewRecord:
    stem: str
    image_sha256: str
    revision: int
    status: ReviewStatus
    anchors: tuple[AnchorPrediction, ...]
    interference_tags: tuple[str, ...]
    exclusion_reason: str | None
    warnings: tuple[str, ...]
    suggestion_modified: bool
    first_reviewed_at: str | None
    second_review_required: bool
    second_reviewed_at: str | None
    notes: str


def record_from_dict(data: dict[str, Any]) -> ReviewRecord:
    return ReviewRecord(
        stem=data["stem"],
        image_sha256=data["image_sha256"],
        revision=data["revision"],
        status=ReviewStatus(data["status"]),
        anchors=tuple(AnchorPrediction(**anchor) for anchor in data["anchors"]),
        interference_tags=tuple(data["interference_tags"]),
        exclusion_reason=data["exclusion_reason"],
        warnings=tuple(data["warnings"]),
        suggestion_modified=data["suggestion_modified"],
        first_reviewed_at=data["first_reviewed_at"],
        second_review_required=data["second_review_required"],
        second_reviewed_at=data["second_reviewed_at"],
        notes=data["notes"],
    )


def record_to_dict(record: ReviewRecord) -> dict[str, Any]:
    return {
        "stem": record.stem,
        "image_sha256": record.image_sha256,
        "revision": record.revision,
        "status": record.status.value,
        "anchors": [anchor.to_dict() for anchor in record.anchors],
        "interference_tags": list(record.interference_tags),
        "exclusion_reason": record.exclusion_reason,
        "warnings": list(record.warnings),
        "suggestion_modified": record.suggestion_modified,
        "first_reviewed_at": record.first_reviewed_at,
        "second_review_required": record.second_review_required,
        "second_reviewed_at": record.second_reviewed_at,
        "notes": record.notes,
    }


def validate_record(record: ReviewRecord) -> list[str]:
    errors: list[str] = []

    if not record.stem:
        errors.append("stem_empty")
    if re.fullmatch(r"[0-9a-f]{64}", record.image_sha256) is None:
        errors.append("image_sha256_invalid")
    if record.revision < 0:
        errors.append("revision_negative")

    if any(tag not in INTERFERENCE_TAGS for tag in record.interference_tags):
        errors.append("unknown_interference_tag")
    if (
        record.exclusion_reason is not None
        and record.exclusion_reason not in EXCLUSION_REASONS
    ):
        errors.append("unknown_exclusion_reason")

    non_finite_coordinate = False
    out_of_bounds = False
    duplicate_y = False
    descending_y = False
    previous_y: int | float | None = None
    seen_y: set[int | float] = set()
    for anchor in record.anchors:
        if not math.isfinite(anchor.x) or not math.isfinite(anchor.y):
            non_finite_coordinate = True
        elif not (0 <= anchor.x <= 639 and 0 <= anchor.y <= 479):
            out_of_bounds = True
        if anchor.y in seen_y:
            duplicate_y = True
        if previous_y is not None:
            if anchor.y < previous_y:
                descending_y = True
        seen_y.add(anchor.y)
        previous_y = anchor.y

    if non_finite_coordinate:
        errors.append("anchor_coordinate_non_finite")
    if out_of_bounds:
        errors.append("anchor_out_of_bounds")
    if duplicate_y:
        errors.append("duplicate_anchor_y")
    if descending_y:
        errors.append("anchor_y_not_increasing")

    if record.status is ReviewStatus.POSITIVE and len(record.anchors) < 3:
        errors.append("positive_requires_three_anchors")
    if record.status is ReviewStatus.HARD_NEGATIVE:
        if record.anchors:
            errors.append("hard_negative_has_anchors")
        if not any(tag in INTERFERENCE_TAGS for tag in record.interference_tags):
            errors.append("hard_negative_requires_interference_tag")

    if record.status is ReviewStatus.EXCLUDED:
        if record.exclusion_reason not in EXCLUSION_REASONS:
            errors.append("excluded_requires_exclusion_reason")
    elif record.exclusion_reason is not None:
        errors.append("non_excluded_has_exclusion_reason")

    if record.status is ReviewStatus.UNREVIEWED:
        if record.first_reviewed_at is not None or record.second_reviewed_at is not None:
            errors.append("unreviewed_has_review_timestamp")
    elif record.status in {
        ReviewStatus.POSITIVE,
        ReviewStatus.HARD_NEGATIVE,
        ReviewStatus.EXCLUDED,
    } and record.first_reviewed_at is None:
        errors.append("final_status_requires_first_review")

    if record.second_reviewed_at is not None and not record.second_review_required:
        errors.append("second_reviewed_without_requirement")

    return errors
