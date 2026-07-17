# Low-Light Human Annotation Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a loopback-only annotation workbench that lets the user manually classify and verify all 388 low-light hose candidates, exports a validated but unmerged `pic+label` bundle, and provides a LabelMe interchange fallback.

**Architecture:** Keep immutable candidates separate from mutable review state. Generate model-assisted suggestions, expose review operations through a small Werkzeug WSGI API and static canvas UI, persist every decision atomically with revision history, then validate and transactionally export only approved positive and hard-negative samples. LabelMe exchanges the same state schema and never bypasses project validation.

**Tech Stack:** Python 3.10.20, Werkzeug 3.0.6, NumPy 1.24.4, OpenCV 4.9, Pillow 10.4, PyTorch, standard browser HTML/CSS/JavaScript, `unittest`.

## Global Constraints

- Bind only to `127.0.0.1`; try ports `8765` through `8799` in order.
- Treat all 388 candidates as requiring an explicit first-pass human decision.
- The target is the hose the robot must physically follow and rewind, independent of color.
- Short, unambiguous occlusions may be interpolated; long or ambiguous occlusions must be reviewed again or excluded.
- Original candidate JPEGs are canonical training images; enhanced images are preview-only.
- System predictions are suggestions and can never set `first_reviewed_at` or approve a sample.
- Hard negatives generate all-zero labels; excluded samples generate no training pair.
- Do not restore the quarantined 60 pairs, merge `datasets/newdata`, train, or export ONNX.
- Keep LabelMe as a fallback interchange path; do not install it unless the primary page fails and the user authorizes installation.
- The workspace is not a Git repository. Do not initialize Git; replace commit steps with a reviewer checkpoint listing changed files and fresh test output.

---

## File Structure

- Create `low_light_dataset/review_models.py`: enums, record schema, serialization, record-level validation.
- Create `low_light_dataset/review_store.py`: manifest loading, state initialization, optimistic revisions, atomic state writes, append-only history.
- Create `low_light_dataset/preannotation.py`: current-model/color/enhanced prediction fusion and temporal disagreement warnings.
- Create `low_light_dataset/review_service.py`: loopback WSGI app, JSON API, media/static routes, port selection and health response.
- Create `low_light_dataset/review_export.py`: second-review selection, full validation, overlays, contact sheets, staging export.
- Create `low_light_dataset/labelme_bridge.py`: project-state to LabelMe JSON and validated LabelMe JSON import.
- Create `low_light_dataset/review_web/index.html`: workbench structure.
- Create `low_light_dataset/review_web/app.js`: API client, canvas editing, state transitions, navigation and autosave.
- Create `low_light_dataset/review_web/styles.css`: responsive local workbench presentation.
- Create `scripts/run_low_light_annotation_workbench.py`: `prepare`, `serve`, `validate`, `export`, `labelme-export`, `labelme-import` commands.
- Create `tests/test_review_models.py`, `tests/test_review_store.py`, `tests/test_preannotation.py`, `tests/test_review_service.py`, `tests/test_review_export.py`, `tests/test_labelme_bridge.py`, and `tests/test_review_cli.py`.
- Modify `操作手册.md`: exact primary and fallback commands, state meanings, shutdown and recovery procedure.

## Stable Interfaces

```python
# low_light_dataset/review_models.py
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

# low_light_dataset/review_store.py
class ReviewStore:
    def initialize(self, manifest_path: Path) -> dict[str, Any]: ...
    def summary(self) -> dict[str, int]: ...
    def get(self, stem: str) -> ReviewRecord: ...
    def update(self, stem: str, patch: dict[str, Any], expected_revision: int,
               actor: str = "human") -> ReviewRecord: ...

# low_light_dataset/preannotation.py
def build_preannotations(candidate_root: Path, checkpoint_path: Path,
                         output_path: Path, device: str = "cuda") -> Path: ...

# low_light_dataset/review_service.py
def create_app(candidate_root: Path, work_root: Path) -> Callable: ...
def find_available_port(host: str = "127.0.0.1", start: int = 8765,
                        end: int = 8799) -> int: ...

# low_light_dataset/review_export.py
def assign_second_review_requirements(work_root: Path, seed: int = 20260716) -> dict[str, Any]: ...
def validate_review(candidate_root: Path, work_root: Path, dataset_root: Path,
                    expected_count: int = 388) -> ValidationReport: ...
def export_reviewed_bundle(candidate_root: Path, work_root: Path,
                           dataset_root: Path, output_root: Path,
                           expected_count: int = 388) -> Path: ...
```

---

### Task 1: Review Domain Model and Record Validation

**Files:**
- Create: `low_light_dataset/review_models.py`
- Test: `tests/test_review_models.py`

**Interfaces:**
- Consumes: `AnchorPrediction` and `regularize_anchors` from `low_light_dataset.annotation`.
- Produces: `ReviewStatus`, `ReviewRecord`, `record_from_dict`, `record_to_dict`, and `validate_record` from the Stable Interfaces section.

- [ ] **Step 1: Write failing record-contract tests**

```python
class ReviewRecordTests(unittest.TestCase):
    def test_positive_requires_three_ordered_in_bounds_anchors(self):
        record = make_record(
            status=ReviewStatus.POSITIVE,
            anchors=(anchor(100, 300), anchor(200, 310)),
            first_reviewed_at="2026-07-16T12:00:00+00:00",
        )
        self.assertIn("positive_requires_three_anchors", validate_record(record))

    def test_hard_negative_requires_tag_and_no_anchors(self):
        record = make_record(
            status=ReviewStatus.HARD_NEGATIVE,
            anchors=(anchor(100, 300),),
            interference_tags=(),
            first_reviewed_at="2026-07-16T12:00:00+00:00",
        )
        self.assertEqual(
            {"hard_negative_must_have_no_anchors", "hard_negative_requires_tag"},
            set(validate_record(record)),
        )

    def test_excluded_requires_known_reason(self):
        record = make_record(status=ReviewStatus.EXCLUDED, exclusion_reason=None)
        self.assertIn("excluded_requires_reason", validate_record(record))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_models -v`

Expected: import failure for `low_light_dataset.review_models`.

- [ ] **Step 3: Implement immutable records, exact allowed values, JSON conversion, and validation**

Implement these constants exactly:

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

`validate_record` must reject unknown tags/reasons, non-increasing anchor `y`, duplicate rows, coordinates outside 640×480, unreviewed records with review timestamps, final records without first-review timestamps, hard negatives with anchors, and positive records with fewer than three anchors.

- [ ] **Step 4: Run focused and existing annotation tests**

Run:
`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_models tests.test_low_light_annotation tests.test_low_light_artifacts -v`

Expected: all tests pass.

- [ ] **Step 5: Reviewer checkpoint**

Record `low_light_dataset/review_models.py`, `tests/test_review_models.py`, and the passing test count in the implementation log.

---

### Task 2: Atomic Review Store and Audit History

**Files:**
- Create: `low_light_dataset/review_store.py`
- Test: `tests/test_review_store.py`

**Interfaces:**
- Consumes: `ReviewRecord`, `record_from_dict`, `record_to_dict`, `validate_record`.
- Produces: `ReviewStore` with `initialize`, `summary`, `get`, and `update`.

- [ ] **Step 1: Write failing initialization, revision-conflict, history, and recovery tests**

```python
def test_initialize_covers_manifest_and_preserves_hashes(self):
    store = ReviewStore(self.work_root, self.candidate_root)
    document = store.initialize(self.manifest)
    self.assertEqual(3, len(document["records"]))
    self.assertEqual(self.rows[0]["image_sha256"], document["records"][0]["image_sha256"])

def test_update_is_atomic_and_rejects_stale_revision(self):
    updated = self.store.update(self.stem, {"status": "needs_second_review"}, 0)
    self.assertEqual(1, updated.revision)
    with self.assertRaises(RevisionConflict):
        self.store.update(self.stem, {"notes": "stale"}, 0)
    self.assertEqual(1, len(self.history_lines()))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v`

Expected: import failure for `low_light_dataset.review_store`.

- [ ] **Step 3: Implement state schema and atomic persistence**

Use `schema_version: 2`, canonical sorted stem order, `annotation_state.json.tmp` followed by `Path.replace`, and append one UTF-8 JSON object per successful update to `annotation_history.jsonl`. Keep `annotation_state.json.last-good` before replacement. `update` must verify the supplied revision, merge only the allowed editable fields, call `validate_record`, increment revision once, write state, then append history. Do not append history if state writing fails.

- [ ] **Step 4: Add candidate-hash guard and last-good recovery**

On `get` and `update`, compute the current JPEG SHA-256 and compare it to the manifest value. Raise `CandidateChangedError(stem)` on mismatch. If current state JSON cannot be decoded, load `.last-good`, leave the corrupt file untouched, and report `recovered_from_last_good: true` in `summary()`.

- [ ] **Step 5: Run focused tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v`

Expected: all tests pass.

- [ ] **Step 6: Reviewer checkpoint**

Record the two new files and focused test output.

---

### Task 3: Model-Assisted Suggestions and Temporal Warnings

**Files:**
- Create: `low_light_dataset/preannotation.py`
- Test: `tests/test_preannotation.py`

**Interfaces:**
- Consumes: `HoseAnnotator`, `extract_color_anchors`, `fuse_predictions`, `regularize_anchors`, `enhance_low_light`, and candidate `manifest.csv`.
- Produces: `build_preannotations(...) -> Path` writing `preannotation.json` without changing review status.

- [ ] **Step 1: Write failing tests with fake annotator outputs**

```python
def test_preannotation_never_marks_a_record_reviewed(self):
    output = build_with_fake_predictor(self.candidate_root, self.output)
    record = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    self.assertNotIn("status", record)
    self.assertNotIn("first_reviewed_at", record)

def test_neighbor_disagreement_adds_warning_without_moving_manual_points(self):
    records = make_three_suggestions(center_x=(120, 500, 125))
    checked = add_temporal_warnings(records, threshold_px=96.0)
    self.assertIn("temporal_disagreement", checked[1]["warnings"])
    self.assertEqual(500.0, checked[1]["anchors"][0]["x"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation -v`

Expected: import failure for `low_light_dataset.preannotation`.

- [ ] **Step 3: Implement per-image suggestion generation**

For each manifest row, load the original JPEG, calculate the preview enhancement in memory, predict on original and preview, extract color anchors on both, fuse original/preview model anchors at 48 px, fuse original/preview color anchors at 64 px, then fuse model/color anchors at 96 px. Regularize the result and write anchors, warnings, source metrics, checkpoint SHA-256, and image SHA-256. Never write enhanced JPEGs into the candidate directory.

- [ ] **Step 4: Implement temporal disagreement warnings**

Compare a record with the nearest earlier and later candidates. At shared anchor rows, flag `temporal_disagreement` when its median x residual from the neighbor interpolation exceeds 96 px. Flag `insufficient_anchor_count` when fewer than three anchors remain. Warnings change review priority only, not anchor geometry or review state.

- [ ] **Step 5: Run focused tests and a two-image CPU smoke test**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_preannotation -v`

Run the smoke fixture through `build_preannotations(..., device="cpu")` and assert the JSON contains exactly the fixture stems.

Expected: all tests and smoke assertion pass.

- [ ] **Step 6: Reviewer checkpoint**

Record the new module, tests, checkpoint path `logs/0716_0745_lr_1e-05_b_8/best_model.pth`, and smoke output.

---

### Task 4: Loopback WSGI Service and Port Fallback

**Files:**
- Create: `low_light_dataset/review_service.py`
- Test: `tests/test_review_service.py`

**Interfaces:**
- Consumes: `ReviewStore`, `enhance_low_light`, static assets in `review_web`.
- Produces: `create_app` and `find_available_port`.

- [ ] **Step 1: Write failing service tests**

```python
def test_health_reports_candidate_and_state_counts(self):
    response = self.client.get("/health")
    self.assertEqual(200, response.status_code)
    self.assertEqual({"ok": True, "candidate_count": 3, "record_count": 3}, response.json)

def test_patch_requires_matching_revision(self):
    response = self.client.patch(
        f"/api/records/{self.stem}",
        json={"revision": 99, "status": "needs_second_review"},
    )
    self.assertEqual(409, response.status_code)

def test_port_selection_skips_occupied_8765(self):
    with listening_socket("127.0.0.1", 8765):
        self.assertEqual(8766, find_available_port())
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Expected: import failure for `low_light_dataset.review_service`.

- [ ] **Step 3: Implement exact routes**

Implement `GET /health`, `GET /api/summary`, `GET /api/records`, `GET /api/records/<stem>`, `PATCH /api/records/<stem>`, `GET /api/review-queue`, `GET /media/original/<stem>.jpg`, `GET /media/enhanced/<stem>.jpg`, and `GET /static/<name>`. Reject path traversal using exact manifest stems and `Path.name == requested_name`. Return JSON 400 for record validation, 404 for unknown stems, 409 for revisions, and 500 with a stable error code for save failures.

For a human PATCH that changes status to `positive`, `hard_negative`, or `excluded`, the service must set `first_reviewed_at` to the current UTC ISO-8601 time when it is still null. A later explicit second-review action sets `second_reviewed_at` server-side. The client, preannotator, LabelMe JSON, and ordinary draft autosaves cannot supply or overwrite either timestamp.

- [ ] **Step 4: Implement preview-only enhanced caching**

Store generated previews under `<work_root>/enhanced_preview_cache/<stem>.jpg`, with a sidecar containing source SHA-256 and enhancement parameters. Regenerate on hash mismatch. The exporter must never read this cache.

- [ ] **Step 5: Implement port scan and explicit loopback enforcement**

`find_available_port` opens a temporary TCP socket on `127.0.0.1` for every port in order. Raise `RuntimeError("no_loopback_port_available_8765_8799")` after 8799. Reject a CLI host other than the literal `127.0.0.1`.

- [ ] **Step 6: Run focused tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Expected: all tests pass.

- [ ] **Step 7: Reviewer checkpoint**

Record the new module, tests, and port fallback result.

---

### Task 5: Browser Annotation UI

**Files:**
- Create: `low_light_dataset/review_web/index.html`
- Create: `low_light_dataset/review_web/app.js`
- Create: `low_light_dataset/review_web/styles.css`
- Test: extend `tests/test_review_service.py`

**Interfaces:**
- Consumes: Task 4 JSON routes and 640×480 original/enhanced media.
- Produces: human-editable anchors and explicit record patches with optimistic revisions.

- [ ] **Step 1: Write failing static-contract tests**

```python
def test_index_exposes_required_controls(self):
    html = self.client.get("/").text
    for element_id in (
        "hose-canvas", "preview-toggle", "previous-frame", "next-frame",
        "mark-positive", "mark-hard-negative", "mark-excluded",
        "mark-needs-review", "undo", "redo", "save-status",
    ):
        self.assertIn(f'id="{element_id}"', html)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Expected: failure because `/` or required controls are missing.

- [ ] **Step 3: Build the single-page layout**

Create a progress header, filter row, main 4:3 canvas, original/enhanced toggle, previous/current/next timestamps, anchor toolbar, classification controls, required tag/reason controls, warnings, notes, revision indicator, and save result. Use responsive CSS that keeps the canvas visible at 1280×720 and stacks controls below 760 px width.

- [ ] **Step 4: Implement coordinate-safe canvas editing**

Convert pointer coordinates with:

```javascript
const x = Math.max(0, Math.min(639, (event.clientX - rect.left) * 640 / rect.width));
const y = Math.max(0, Math.min(479, (event.clientY - rect.top) * 480 / rect.height));
```

Keep anchors sorted by `y`; dragging within six display pixels selects an anchor, Delete removes it, Ctrl+Z/Ctrl+Y operate bounded undo/redo stacks, and clear requires a confirmation dialog. Draw suggestions in amber and current anchors/line in red. Do not implement a one-key final approval shortcut.

- [ ] **Step 5: Implement classification validation and autosave**

Positive requires at least three anchors; hard negative requires at least one interference tag and clears anchors only after confirmation; excluded requires one exclusion reason; needs-review accepts notes and warnings. Debounce draft anchor autosaves by 500 ms. Final classification buttons display the chosen class in a confirmation dialog and PATCH with the current revision. On 409, reload the record and show a revision-conflict message without overwriting local unsaved anchors.

- [ ] **Step 6: Add adjacent-frame and enhanced-preview behavior**

Previous/next thumbnails use chronological manifest neighbors. The preview toggle changes only the background image URL, not anchor coordinates. Display a permanent banner: `增强图仅辅助观察，导出始终使用原图`.

- [ ] **Step 7: Run service/static tests and manually inspect one synthetic record**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Start a three-record fixture, open the reported loopback URL, add/move/delete anchors, toggle preview, set `needs_second_review`, reload, and verify the state survives.

Expected: tests pass and reload preserves the last saved revision.

- [ ] **Step 8: Reviewer checkpoint**

Record the three assets, test output, actual loopback URL, and manual smoke observations.

---

### Task 6: LabelMe Interchange Fallback

**Files:**
- Create: `low_light_dataset/labelme_bridge.py`
- Test: `tests/test_labelme_bridge.py`

**Interfaces:**
- Consumes: `ReviewStore`, candidate images, stable tag/reason sets.
- Produces: `export_labelme(candidate_root, work_root, output_root) -> Path` and `import_labelme(input_root, store) -> dict[str, Any]`.

- [ ] **Step 1: Write failing LabelMe round-trip and rejection tests**

```python
def test_positive_round_trip_preserves_centerline_points(self):
    exported = export_labelme(self.candidates, self.work, self.labelme)
    document = json.loads((exported / f"{self.stem}.json").read_text("utf-8"))
    self.assertEqual("target_hose_centerline", document["shapes"][0]["label"])
    imported = import_labelme(exported, self.store)
    self.assertEqual(1, imported["updated"])

def test_import_rejects_multiple_target_centerlines(self):
    write_labelme_json(shapes=[centerline(), centerline()])
    with self.assertRaisesRegex(ValueError, "multiple_target_centerlines"):
        import_labelme(self.labelme, self.store)
```

- [ ] **Step 2: Run focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_labelme_bridge -v`

Expected: import failure for `low_light_dataset.labelme_bridge`.

- [ ] **Step 3: Implement exact LabelMe mapping**

Use `version: "5.x"`, `imageData: null`, and relative `imagePath`. Positive uses exactly one `shape_type: "linestrip"` named `target_hose_centerline`. Store status in flags named `status_positive`, `status_hard_negative`, `status_excluded`, or `status_needs_second_review`; store tags as `interference_<tag>` and exclusion as `exclude_<reason>`. Reject no status, multiple statuses, unknown flags, unknown shape labels, more than one target line, positive lines with fewer than three points, or non-positive records containing a target line.

- [ ] **Step 4: Import through the store, not by editing state files**

Each valid JSON calls `ReviewStore.update` with the current revision and actor `labelme_import`. Validate all input files first; if any input is invalid, report every file error and perform zero updates.

- [ ] **Step 5: Run focused tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_labelme_bridge -v`

Expected: all tests pass without installing LabelMe.

- [ ] **Step 6: Reviewer checkpoint**

Record bridge files, test output, and a sample exported JSON path.

---

### Task 7: Second Review Assignment, Validation, and Transactional Export

**Files:**
- Create: `low_light_dataset/review_export.py`
- Test: `tests/test_review_export.py`

**Interfaces:**
- Consumes: `ReviewStore`, `validate_record`, `rasterize_centerline`, `save_label`, `save_overlay`, formal dataset snapshot utilities.
- Produces: `assign_second_review_requirements`, `validate_review`, and `export_reviewed_bundle`.

- [ ] **Step 1: Write failing second-review tests**

```python
def test_all_hard_negatives_and_flagged_positives_require_second_review(self):
    report = assign_second_review_requirements(self.work, seed=20260716)
    self.assertIn("hard_negative", report["required_reasons"][self.negative_stem])
    self.assertIn("suggestion_modified", report["required_reasons"][self.modified_stem])

def test_plain_positives_receive_deterministic_ten_percent_audit(self):
    left = assign_second_review_requirements(self.work, seed=20260716)
    right = assign_second_review_requirements(self.work, seed=20260716)
    self.assertEqual(left["random_audit_stems"], right["random_audit_stems"])
    self.assertEqual(math.ceil(len(self.plain_stems) * 0.10), len(left["random_audit_stems"]))
```

- [ ] **Step 2: Write failing validation/export tests**

Test incomplete 388 coverage, pending second review, source hash change, non-original export hash, test-set hash leakage, hard-negative nonzero label, excluded sample leakage, stale staging directory, and successful paired export.

- [ ] **Step 3: Run focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_export -v`

Expected: import failure for `low_light_dataset.review_export`.

- [ ] **Step 4: Implement deterministic review assignment**

Always require all hard negatives; positives with `suggestion_modified`, `temporal_disagreement`, prediction conflicts, occlusion warnings, or interference tags; and every record that previously held needs-review according to history. From remaining positive stems, order by SHA-256 of `f"20260716:{stem}"` and select the first `ceil(n * 0.10)`.

- [ ] **Step 5: Implement full validation**

Require exactly `expected_count` records, no unreviewed/needs-review records, complete first review, complete required second review, source hashes matching manifest, allowed record-level state, image size/mode, positive nonempty labels, hard-negative empty labels, no excluded pairs, paired stems, no formal test-image hash collision, and no candidate filename collision with formal train/test stems.

- [ ] **Step 6: Implement staging export**

Refuse to overwrite existing output or `<output>.staging`. Copy original candidate JPEGs for positive and hard-negative records and verify their SHA-256. Rasterize positives and generate all-zero hard-negative labels. Generate overlays, 20-item contact sheets, `annotation.json`, `review_report.json`, and `validation_report.json`. Re-run validation against staging and atomically rename staging to the final output only when `ok` is true.

- [ ] **Step 7: Run focused tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_export -v`

Expected: all tests pass.

- [ ] **Step 8: Reviewer checkpoint**

Record module/tests and successful temporary export counts.

---

### Task 8: CLI, Operational Documentation, and Full Verification

**Files:**
- Create: `scripts/run_low_light_annotation_workbench.py`
- Create: `tests/test_review_cli.py`
- Modify: `操作手册.md`

**Interfaces:**
- Consumes: all modules from Tasks 1–7.
- Produces: stable operator commands and end-to-end verification.

- [ ] **Step 1: Write failing parser and help tests**

```python
def test_commands_parse(self):
    parser = build_parser()
    for command in ("prepare", "serve", "validate", "export", "labelme-export", "labelme-import"):
        self.assertEqual(command, parser.parse_args([command, *minimal_args(command)]).command)

def test_help_runs_from_project_root(self):
    result = subprocess.run([PYTHON, "scripts/run_low_light_annotation_workbench.py", "--help"], capture_output=True, text=True)
    self.assertEqual(0, result.returncode)
```

- [ ] **Step 2: Run focused test and verify RED**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_cli -v`

Expected: script import or command failure.

- [ ] **Step 3: Implement exact CLI behavior**

`prepare` initializes state and creates preannotations using
`logs/0716_0745_lr_1e-05_b_8/best_model.pth`; `serve` scans ports and runs the WSGI app; `validate` prints the full report; `export` targets
`datasets/low_light_camera_full_rgb_v2_reviewed`; LabelMe commands call the bridge. Every command prints UTF-8 JSON containing `ok`, paths, counts, and errors. `export` exits nonzero unless validation passes. No command exposes merge or train behavior.

- [ ] **Step 4: Add the exact primary operating commands to the manual**

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py prepare --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work --checkpoint logs\0716_0745_lr_1e-05_b_8\best_model.pth --device cuda
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py serve --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py validate --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work --dataset datasets\newdata
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py export --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work --dataset datasets\newdata --output datasets\low_light_camera_full_rgb_v2_reviewed
```

Document that the URL printed by `serve` is authoritative and can be `8765`–`8799`.

- [ ] **Step 5: Add exact LabelMe fallback commands to the manual**

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py labelme-export --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work --output datasets\low_light_camera_full_rgb_v2_labelme
D:\Anaconda3\envs\hosebot_cv\python.exe scripts\run_low_light_annotation_workbench.py labelme-import --input datasets\low_light_camera_full_rgb_v2_labelme --candidates datasets\low_light_camera_full_rgb_v2_candidates --work datasets\low_light_camera_full_rgb_v2_review_work
```

State that installing or launching LabelMe requires a separate user authorization only if the primary page fails.

- [ ] **Step 6: Run the complete test suite**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v`

Expected: all existing 54 tests plus all new review-workbench tests pass with zero failures and zero errors.

- [ ] **Step 7: Run production read-only preflight before touching review state**

Verify candidate count 388, unique stems 388, all image hashes match the manifest, dimensions are 640×480, candidate PNG count is zero, formal train count remains 4858/4858, formal test count remains 867/867, and quarantined old-prefix residue in formal train is zero.

- [ ] **Step 8: Start the real primary service and verify health**

Run `prepare`, then `serve`. Read the printed URL, request `<url>/health`, and require JSON `ok: true`, `candidate_count: 388`, and `record_count: 388`. Manually load the page, verify one draft anchor edit persists after reload, then restore that record to `unreviewed` so implementation verification does not count as human annotation.

- [ ] **Step 9: Stop at the human-review handoff**

Report the actual URL, work-state paths, test count, health output, and LabelMe fallback directory. Do not classify any of the 388 samples on the user's behalf, do not export the reviewed bundle, do not merge data, and do not train.

- [ ] **Step 10: Reviewer checkpoint**

List every created/modified file and attach the fresh full-suite and production-preflight outputs.

---

## Execution Completion Boundary

Executing this plan is complete when the primary workbench is implemented, tested, serving a healthy loopback URL, and ready for the user's 388-image review. Human classification, second review, reviewed-bundle export, formal merge, and training are later interactive phases and are not implementation-completion criteria.
