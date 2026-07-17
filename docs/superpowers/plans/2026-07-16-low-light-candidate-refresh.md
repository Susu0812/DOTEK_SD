# Low-Light Candidate Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely quarantine the flawed 60-pair low-light merge and build an auditable, event-enriched unlabeled candidate set from the first five minutes of `camera_full_rgb.mp4`.

**Architecture:** Add one focused module for transactional quarantine and one focused module for video candidate analysis/extraction. Expose both through a new non-training CLI so the existing dataset builder remains backward compatible. All source mutations are hash-gated and all generated artifacts carry receipts.

**Tech Stack:** Python 3.10, OpenCV, NumPy, Pillow, SciPy-free image metrics, unittest.

## Global Constraints

- Never delete the historical 60-pair bundle.
- Do not create labels or launch training.
- Quarantine must restore train/test counts to 4858/867 and preserve pairing.
- Candidate analysis is limited to `[0, 300)` seconds at a 0.25-second analysis interval.
- Base one-second frames are mandatory; event frames are supplemental and de-duplicated.
- The workspace is not a Git repository, so commit steps are replaced by file/hash verification.

---

### Task 1: Transactional quarantine

**Files:**
- Create: `low_light_dataset/quarantine.py`
- Create: `tests/test_low_light_quarantine.py`

**Interfaces:**
- Produces: `quarantine_merged_bundle(bundle_root: Path, dataset_root: Path, quarantine_root: Path, expected_count: int = 60, expected_train_after: int = 4858, expected_test_count: int = 867) -> Path`

- [ ] Write tests covering hash mismatch refusal, successful 1-pair quarantine, final pairing/count validation, and idempotent re-run.
- [ ] Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_low_light_quarantine -v`; expect failures because the module is absent.
- [ ] Implement receipt parsing, SHA-256 preflight, staging copy verification, source unlink only after verified copy, atomic staging rename, and post-count checks.
- [ ] Re-run the focused tests; expect all PASS.

### Task 2: Candidate sampling and artifact generation

**Files:**
- Create: `low_light_dataset/candidates.py`
- Create: `tests/test_low_light_candidates.py`

**Interfaces:**
- Produces: `extract_candidate_set(video_path: Path, output_root: Path, duration_seconds: float = 300.0, analysis_interval_seconds: float = 0.25) -> Path`
- Produces: `CandidateMetrics`, `CandidateRecord`, `difference_hash(image_bgr: np.ndarray) -> int`, and `hamming_distance(left: int, right: int) -> int`.

- [ ] Write synthetic-video tests for mandatory one-second base frames, at-most-one event frame per second, no labels, dHash de-duplication, manifest fields, 640×480 output, and refusal to overwrite.
- [ ] Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_low_light_candidates -v`; expect failures because the module is absent.
- [ ] Implement sequential video decoding, quarter-second frame selection, frame-difference/flow/brightness/sharpness metrics, per-second event ranking, dHash de-duplication, JPEG writing, CSV/JSON receipts, and 20-image contact sheets.
- [ ] Re-run the focused tests; expect all PASS.

### Task 3: Operational CLI

**Files:**
- Create: `scripts/refresh_low_light_candidates.py`
- Create: `tests/test_low_light_refresh_cli.py`

**Interfaces:**
- Command: `quarantine --bundle <path> --dataset <path> --output <path>`
- Command: `extract --video <path> --output <path> --duration 300 --interval 0.25`
- Command: `verify --dataset <path> --quarantine <path> --candidates <path>`

- [ ] Write parser and subprocess help tests for all three commands.
- [ ] Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_low_light_refresh_cli -v`; expect failures because the script is absent.
- [ ] Implement the minimal CLI that calls the two modules and prints UTF-8 JSON.
- [ ] Re-run the focused tests; expect all PASS.

### Task 4: Execute quarantine and extraction

**Files generated:**
- `datasets/quarantine/low_light_camera_full_rgb_v1/**`
- `datasets/low_light_camera_full_rgb_v2_candidates/**`

- [ ] Run quarantine against the real merge receipt and formal dataset.
- [ ] Verify train/test counts, stem pairing, receipt hashes, and absence of the 60 stems from formal train.
- [ ] Run candidate extraction against `E:\New_DOTEK\数据\camera_full_rgb.mp4`.
- [ ] Run the CLI verifier and confirm no label files and no formal dataset contamination.
- [ ] Open the first, middle, and last contact sheet for visual inspection.

### Task 5: Regression verification

**Files:** none beyond test caches.

- [ ] Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v` and require all tests to pass.
- [ ] Record SHA-256 for the quarantine receipt, candidate summary, and manifest.
- [ ] Report exact output paths, counts, event/base distribution, and any frames rejected during decoding.

