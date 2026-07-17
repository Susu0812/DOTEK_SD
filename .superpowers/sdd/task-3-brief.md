# Task 3 Brief: Model-Assisted Suggestions and Temporal Warnings

Implement only Task 3 of the approved annotation-workbench plan. Follow strict test-driven development: create `tests/test_preannotation.py`, demonstrate the expected import/behavior RED, then create `low_light_dataset/preannotation.py`, run focused GREEN and the requested CPU smoke. Do not modify review status, candidates, datasets, training code, checkpoints, or later-task files.

## Inputs and fixed production facts

- Candidate root: `datasets/low_light_camera_full_rgb_v2_candidates`
- Candidate images: `candidate_root/frames/<stem>.jpg`
- Manifest: `candidate_root/manifest.csv`, currently 388 records with `stem`, `target_timestamp_seconds`, and `image_sha256` among other metadata.
- Fixed checkpoint for the later production run: `logs/0716_0745_lr_1e-05_b_8/best_model.pth`
- Reuse the existing public interfaces exactly where applicable:
  - `HoseAnnotator(checkpoint_path, device=device).predict(frame_bgr)`
  - `extract_color_anchors(frame_bgr)`
  - `fuse_predictions(original, enhanced, max_disagreement_px=...)`
  - `regularize_anchors(predictions)`
  - `enhance_low_light(frame_bgr)`; its enhanced image is memory-only.

## Required public interfaces

Implement at minimum:

```python
def build_preannotations(
    candidate_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> Path:
    ...

def add_temporal_warnings(
    records: list[dict[str, Any]],
    threshold_px: float = 96.0,
) -> list[dict[str, Any]]:
    ...
```

Private dependency injection or patchable module symbols are encouraged so tests do not load the 98 MB checkpoint.

## Per-image suggestion pipeline

For each manifest row in numeric `target_timestamp_seconds` order:

1. Validate required manifest fields, duplicate stems, real JPEG existence, and the JPEG SHA-256 against `image_sha256` before accepting the record.
2. Read the original JPEG as BGR. Reject unreadable or non-640x480 candidates; never rewrite it.
3. Compute `enhance_low_light(original)` in memory only. Never create an enhanced JPEG under the candidate root or use enhanced output as a training image.
4. Predict model anchors on original and enhanced images.
5. Extract color anchors on original and enhanced images.
6. Fuse original/enhanced model anchors at 48 px.
7. Fuse original/enhanced color anchors at 64 px.
8. Fuse the model/color results at 96 px.
9. Regularize the combined anchors.
10. Preserve all source/fusion/regularization warnings as a deterministic unique list. Ensure `insufficient_anchor_count` is present whenever fewer than three final anchors remain.

Each output record must contain at least:

- `stem`
- numeric `target_timestamp_seconds`
- manifest `image_sha256`
- final regularized `anchors` serialized with y/x/confidence/source
- `warnings`
- deterministic `source_metrics` sufficient to audit raw/enhanced model counts, raw/enhanced color counts, fused counts, final count, and enhancement before/after metrics and parameters.

The output document must contain schema version, checkpoint SHA-256, and exactly one record for every manifest stem. It must never contain human workflow fields such as `status`, `first_reviewed_at`, `second_reviewed_at`, or `revision`. Write deterministic UTF-8 JSON via a same-directory temporary file and atomic replace; return `output_path`.

Reject an `output_path` located inside `candidate_root`, so the candidate tree stays read-only. On any validation or prediction failure, do not publish a partial output file.

## Temporal warnings

- Work on copies; never move, replace, smooth, or otherwise mutate anchor geometry.
- Order by numeric timestamp and compare each interior record with its nearest earlier and later candidates.
- At anchor rows shared by current, earlier, and later records, linearly interpolate expected x in time between the earlier/later x values.
- Compute absolute x residuals for the current record; if their median is strictly greater than `threshold_px`, add `temporal_disagreement`.
- If fewer than three final anchors are present, add `insufficient_anchor_count`.
- Lack of enough shared rows/neighbors does not invent geometry or auto-approve anything.
- Return deterministic warnings with no duplicates.

## Required tests

Use temporary fixtures, real decodable 640x480 JPEGs, and fake annotator/enhancement functions. Cover at minimum:

1. Import RED before implementation.
2. Fusion thresholds are passed as 48, 64, 96 in the correct order; original and enhanced predictions/extractions are both used.
3. Output never contains human state/timestamp/revision fields.
4. Candidate original bytes/hash/tree remain unchanged and no enhanced JPEG is written.
5. Hash mismatch, duplicate/malformed manifest, unreadable/wrong-size JPEG, missing checkpoint, and output-inside-candidate fail before publishing partial JSON.
6. Temporal disagreement is added for a large median residual without changing anchor x values; stable sequences and insufficient shared rows do not get a false disagreement.
7. Fewer than three anchors yields `insufficient_anchor_count`.
8. Deterministic ordering/output and exact fixture stem coverage.
9. A two-image `device="cpu"` smoke fixture through `build_preannotations` asserts exactly the two stems and confirms the fake annotator received `cpu`.

Run:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation -v`

Then run the regression set:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation tests.test_review_store tests.test_review_models -v`

## Handoff

Create `.superpowers/sdd/task-3-report.md` recording:

- RED and final GREEN commands/results/test counts;
- exact files changed;
- JSON schema and temporal-warning behavior;
- checkpoint path that the later production run will use;
- CPU smoke result;
- any residual concern.

Do not run the 388-image production preannotation yet; Task 8 owns production preflight/prepare. Do not merge, export reviewed data, train, or create ONNX artifacts.
