# Task 7 — Reviewed bundle validation and safe export

## Scope

Implement `low_light_dataset/review_export.py` and focused tests in
`tests/test_review_export.py`. Do not merge into `datasets/`, do not train, and do
not mark any candidate as reviewed. The current production review is incomplete;
the exporter must fail closed rather than weakening review requirements.

## Public API

```python
assign_second_review_requirements(
    candidate_root: Path,
    work_root: Path,
    seed: int = 20260716,
) -> dict[str, Any]

validate_review(
    candidate_root: Path,
    work_root: Path,
    dataset_root: Path,
    expected_count: int = 388,
    bundle_root: Path | None = None,
    seed: int = 20260716,
) -> dict[str, Any]

export_reviewed_bundle(
    candidate_root: Path,
    work_root: Path,
    dataset_root: Path,
    output_root: Path,
    expected_count: int = 388,
    seed: int = 20260716,
) -> Path
```

Return structured, deterministic validation reports. Export raises a stable
exception when preflight fails and leaves the formal dataset byte-for-byte
unchanged.

## TDD requirement

Write failing tests first, run them, then implement the minimum production code.
Keep a short implementation/test report in `.superpowers/sdd/task-7-report.md`.

## Deterministic second-review assignment

Require second review for:

- every `hard_negative`;
- every positive with `suggestion_modified`;
- warnings indicating temporal disagreement, prediction/model conflict,
  low-confidence prediction, or occlusion;
- any non-empty interference tags;
- any stem whose audit history ever had status `needs_second_review` or
  `second_review_required=true`;
- a deterministic audit sample of `ceil(10%)` of remaining plain positives,
  ranked by SHA-256 of `f"{seed}:{stem}"`.

Assignment is monotonic and idempotent: set only false→true, never clear a flag
and never overwrite review timestamps/status/anchors. Refuse assignment if state
or audit history is incomplete/corrupt. Since this is an explicit operator
command, publish through `ReviewStore.update` with a non-empty assignment actor
and `origin="human"`; never impersonate completed human review.

The returned report identifies mandatory stems, sampled stems, already-required
stems, newly-required stems, and reasons.

## Pre-export validation

Validation is read-only and reports all detected errors with stable codes.
At minimum enforce:

- record count equals `expected_count` and state identities match the candidate
  manifest;
- no `unreviewed` or `needs_second_review` status;
- every final record has a first-review timestamp;
- every required second review has a second-review timestamp;
- current candidate JPEG hashes match their manifest/state hashes;
- every record passes `validate_record`;
- candidate files are decodable RGB/JPEG images of exactly 640×480;
- positive records rasterize to a non-empty 480×640 uint8 0/1 mask with at most
  one positive pixel per row;
- hard-negative records rasterize/export as an all-zero mask;
- excluded records never appear as exported pairs;
- paired image/label stems are exact when validating a staging bundle;
- candidate stems do not collide with formal train/test image or label stems;
- candidate image hashes do not collide with any formal test image hash;
- audit state is complete and there is no stale temporary state.

Treat missing/malformed files, symlinks/reparse escapes, unsafe stems and paths,
and unreadable JSON/history as validation failures, not crashes that accidentally
permit export. Do not follow candidate or output links outside their roots.

## Transactional export

- Refuse if `output_root` already exists or if `<output_root>.staging` exists.
- Refuse paths nested in/equal to candidate, work, or formal dataset roots.
- Preflight first. Then create `<output_root>.staging`.
- Copy canonical original candidate JPEG bytes for `positive` and
  `hard_negative` only; verify copied SHA-256 equals manifest hash.
- Create positive masks with `rasterize_centerline`; create exact all-zero masks
  for hard negatives; save using `save_label`.
- Create overlays from the canonical original images using `save_overlay`.
- Create deterministic overlay contact sheets with at most 20 items per sheet.
- Write `annotation.json`, `review_report.json`, and `validation_report.json`
  atomically. Reports include exported/excluded counts, stems, hashes, second
  review assignment, and formal dataset before/after snapshots.
- Revalidate the complete staging tree, including original-image copy hashes.
- Confirm the formal dataset snapshot did not change.
- Atomically rename staging to output only after every check passes.
- On failure, never publish `output_root`; leaving staging for forensic recovery
  is acceptable and a later run must refuse it as stale.

Expected bundle layout:

```text
output_root/
  pic/*.jpg
  label/*.png
  overlay/*.jpg
  contact_sheets/overlay_contact_001.jpg
  annotation.json
  review_report.json
  validation_report.json
```

## Tests required

Cover at least:

1. incomplete expected coverage;
2. unreviewed and pending second review;
3. deterministic/monotonic second-review assignment;
4. candidate/source hash change;
5. copied image not identical to original;
6. formal test-set hash leakage;
7. candidate/formal filename collision;
8. hard-negative non-zero label rejection;
9. excluded pair leakage rejection;
10. mismatched paired stems;
11. existing output and stale staging refusal;
12. successful paired export with reports/contact sheet and unchanged formal
    dataset snapshot.

Run focused tests, then the full existing test suite with
`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v`.

