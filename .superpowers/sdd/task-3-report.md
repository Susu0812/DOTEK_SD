# Task 3 Report: Model-Assisted Suggestions and Temporal Warnings

## Status

Implemented Task 3 only with strict test-first development. No 388-image production
preannotation, review-state update, candidate generation, dataset export, training,
checkpoint mutation, or ONNX work was run.

## Implementation summary

- Added manifest-driven preannotation in numeric timestamp order with required-field,
  duplicate-stem, finite-timestamp, lowercase SHA-256, JPEG existence/format/hash,
  decode, and exact 640x480 BGR validation.
- Validates every fixture candidate before constructing the annotator, and revalidates
  each exact JPEG immediately before prediction.
- Runs original/enhanced model and color extraction, fuses model at 48 px, color at
  64 px, model/color at 96 px, then regularizes the combined anchors.
- Preserves all available source, fusion, and regularization warnings as a sorted,
  deterministic unique list; final records with fewer than three anchors always get
  `insufficient_anchor_count`.
- Records schema version 1, checkpoint SHA-256, manifest image SHA-256, numeric target
  timestamp, serialized final anchors, warnings, and deterministic source metrics.
- Keeps enhancement memory-only, rejects an output path inside the candidate root or
  resolved-aliasing the checkpoint, and publishes only a complete document via an
  exclusively created randomized same-directory temporary file plus atomic
  `Path.replace`. Validation or prediction failures clean up only the owned temporary
  file and do not publish partial JSON.
- Adds temporal warnings on deep copies ordered by timestamp. Each interior record
  uses the nearest strictly earlier/later timestamps and at least three rows shared
  by all three records. Expected x is linearly interpolated in time; only a median
  residual strictly greater than the threshold adds `temporal_disagreement`. Anchor
  geometry is never moved, replaced, or smoothed.

## JSON schema

The deterministic UTF-8 document has this top-level shape:

```json
{
  "schema_version": 1,
  "checkpoint_sha256": "<64 lowercase hex characters>",
  "records": [
    {
      "stem": "<manifest stem>",
      "target_timestamp_seconds": 0.0,
      "image_sha256": "<manifest SHA-256>",
      "anchors": [
        {"y": 220, "x": 320.0, "confidence": 0.9, "source": "fused"}
      ],
      "warnings": [],
      "source_metrics": {
        "model": {
          "original_count": 0,
          "enhanced_count": 0,
          "fused_count": 0
        },
        "color": {
          "original_count": 0,
          "enhanced_count": 0,
          "fused_count": 0
        },
        "combined_fused_count": 0,
        "final_count": 0,
        "enhancement": {
          "before": {},
          "after": {},
          "params": {}
        }
      }
    }
  ]
}
```

Human workflow fields (`status`, `first_reviewed_at`, `second_reviewed_at`, and
`revision`) are not emitted.

## Exact files changed

- Created `low_light_dataset/preannotation.py`
- Created `tests/test_preannotation.py`
- Created `.superpowers/sdd/task-3-report.md`

No Task 1/2 source or test file was modified.

## TDD evidence

### RED

The test file was created before the production module.

Command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation -v
```

Observed before `low_light_dataset/preannotation.py` existed:

```text
ImportError: cannot import name 'preannotation' from 'low_light_dataset'
Ran 1 test in 0.001s
FAILED (errors=1)
```

This was the expected import RED caused only by the missing Task 3 module.

### Initial focused GREEN

Same command after the minimal implementation:

```text
Ran 7 tests in 1.582s
OK
```

The seven test methods include eight validation-failure subcases and cover the
required fusion order and thresholds, warning and metrics serialization, prohibited
human fields, candidate-tree byte preservation, deterministic output and exact stem
coverage, validation/prediction no-partial-publication behavior, temporal residuals,
insufficient shared rows, and insufficient anchor counts.

### Initial required regression GREEN

Command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation tests.test_review_store tests.test_review_models -v
```

Observed:

```text
Ran 57 tests in 4.045s
OK
```

### Independent-review filesystem-safety RED

The read-only reviewer found that an output/checkpoint alias could overwrite the
checkpoint and that the predictable `<output>.tmp` path could truncate or delete an
unrelated pre-existing file. Regression tests were added before each production fix.

Checkpoint-alias RED:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation.PreannotationPipelineTests.test_output_cannot_alias_and_overwrite_checkpoint -v
Ran 1 test in 0.197s
FAILED (failures=1: ValueError not raised)
```

Predictable-temporary RED:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation.PreannotationPipelineTests.test_atomic_write_does_not_touch_predictable_preexisting_temp_file tests.test_preannotation.PreannotationPipelineTests.test_replace_failure_keeps_existing_output_and_unrelated_temp_unchanged -v
Ran 2 tests in 0.422s
FAILED (errors=2: unrelated temp sentinel was deleted)
```

Targeted filesystem-safety GREEN after the minimal fixes:

```text
Ran 3 tests in 0.568s
OK
```

The final alias regression covers direct and relative aliases and a symlink alias when
the host permits symlink creation. It proves checkpoint bytes stay unchanged. Atomic
publication now uses `tempfile.NamedTemporaryFile(delete=False, dir=output.parent,
...)`, which creates its randomized path exclusively; success leaves an unrelated
`<output>.tmp` untouched, and replace failure preserves both prior output bytes and
the unrelated sentinel while cleaning the invocation-owned temporary file.

### Final focused GREEN

Fresh command after review remediation:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation -v
Ran 12 tests in 1.857s
OK
```

The expanded coverage also preserves/deduplicates model and color source warnings,
snapshots candidate directories as well as file bytes, checks exact-threshold temporal
behavior, and ensures equal-timestamp records are not selected as earlier/later
neighbors.

### Final required regression GREEN

Fresh required command after all Task 3 production and test changes:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation tests.test_review_store tests.test_review_models -v
Ran 62 tests in 4.931s
OK
```

The independent reviewer then re-read both fixes and reported both prior Important
issues closed, with no new Critical or Important findings.

## CPU smoke

The focused suite runs a real two-image 640x480 JPEG fixture through
`build_preannotations(..., device="cpu")` with fake model/enhancement dependencies.
It asserted exact two-stem coverage in numeric timestamp order and observed the fake
annotator receive `cpu`. The same test builds twice and verifies byte-identical JSON.
Result: passed as part of final focused 12/12 and regression 62/62.

## Later production checkpoint

Task 8's later production run will use:

```text
logs/0716_0745_lr_1e-05_b_8/best_model.pth
```

Task 3 did not load that 98 MB production checkpoint or run the 388-image candidate
set.

## Concerns

- Temporal disagreement requires at least three rows shared by the current, earlier,
  and later records. The brief says insufficient shared rows must not create a false
  warning but does not assign an explicit minimum; three matches the minimum final
  anchor count and is recorded here for Task 8 preflight visibility.
- The implementation intentionally performs a complete image validation pass and a
  second per-image validation immediately before prediction. This increases JPEG I/O
  but avoids retaining roughly hundreds of megabytes of decoded candidate frames and
  ensures prediction never uses bytes that differ from the manifest hash.
