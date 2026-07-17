# Task 1 Brief: Review Domain Model and Record Validation

## Context

This is the first implementation task for the low-light human annotation workbench. It establishes the immutable record contract that all later persistence, API, LabelMe, review, and export tasks depend on.

## Global constraints

- Bind only to `127.0.0.1`; try ports `8765` through `8799` in order.
- Treat all 388 candidates as requiring an explicit first-pass human decision.
- The target is the hose the robot must physically follow and rewind, independent of color.
- Short, unambiguous occlusions may be interpolated; long or ambiguous occlusions must be reviewed again or excluded.
- Original candidate JPEGs are canonical training images; enhanced images are preview-only.
- System predictions are suggestions and can never set `first_reviewed_at` or approve a sample.
- Hard negatives generate all-zero labels; excluded samples generate no training pair.
- Do not restore the quarantined 60 pairs, merge `datasets/newdata`, train, or export ONNX.
- The workspace is not a Git repository. Do not initialize Git or commit.

## Files

- Create: `low_light_dataset/review_models.py`
- Create: `tests/test_review_models.py`

## Required interfaces

```python
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

def record_from_dict(data: dict[str, Any]) -> ReviewRecord: ...
def record_to_dict(record: ReviewRecord) -> dict[str, Any]: ...
def validate_record(record: ReviewRecord) -> list[str]: ...
```

Consume `AnchorPrediction` from `low_light_dataset.annotation`.

Use these allowed values exactly:

```python
INTERFERENCE_TAGS = frozenset({
    "floor_seam_or_edge", "shadow_or_reflection", "cable_or_rope",
    "non_target_hose", "elongated_object", "motion_or_low_light_artifact",
})
EXCLUSION_REASONS = frozenset({
    "near_duplicate", "severe_motion_blur", "severe_exposure_failure",
    "ambiguous_target_identity", "long_occlusion_unknown_path",
    "no_training_value",
})
```

## TDD requirements

1. Write `tests/test_review_models.py` first.
2. Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_models -v` and record the expected RED import failure.
3. Implement the minimal production module.
4. Run the focused test and then:
   `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_models tests.test_low_light_annotation tests.test_low_light_artifacts -v`.
5. Both commands must be recorded in the report with RED and GREEN evidence.

## Required behavior

`validate_record` returns stable error-code strings and must reject:

- unknown interference tags or exclusion reasons;
- negative revision numbers;
- empty stem or non-64-character lowercase hexadecimal SHA-256;
- non-increasing anchor `y`, duplicate anchor rows, non-finite coordinates, and points outside `x=[0,639]`, `y=[0,479]`;
- positive records with fewer than three anchors;
- hard negatives with anchors or without at least one allowed interference tag;
- excluded records without exactly one allowed exclusion reason;
- non-excluded records with an exclusion reason;
- unreviewed records containing first/second review timestamps;
- final `positive`, `hard_negative`, or `excluded` records without `first_reviewed_at`;
- records with `second_reviewed_at` when `second_review_required` is false;
- records requiring second review but marked complete without `second_reviewed_at` only when a separate final-gate caller explicitly asks for completion. Record-level validation must allow the intermediate state where second review is required but not yet completed.

`record_from_dict` must reject missing required fields and invalid enum values, convert anchor dictionaries into `AnchorPrediction`, and preserve confidence/source. `record_to_dict` must emit JSON-compatible lists/dicts with a string status. A round trip must compare equal.

Tests must cover positive, hard negative, excluded, unreviewed, needs-second-review, bad coordinates, bad hashes, unknown values, timestamps, and JSON round trip. Use real records rather than mocking validation internals.

## Completion report

Write the full report to `.superpowers/sdd/task-1-report.md` with:

- status;
- implementation summary;
- RED and GREEN command/output evidence;
- exact changed files;
- self-review findings;
- concerns, including any requirement that could not be met.

Return only the short status summary and report path to the controller.
