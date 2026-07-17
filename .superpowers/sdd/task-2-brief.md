# Task 2 Brief: Atomic Review Store and Audit History

## Context

Task 1 created `ReviewStatus`, immutable `ReviewRecord`, serialization, and validation in `low_light_dataset/review_models.py`. Task 2 builds the only supported persistence boundary for later API, LabelMe, and export code. No caller may edit state JSON directly.

## Global constraints

- All 388 production candidates will eventually require an explicit first-pass human decision.
- Candidate JPEGs and `manifest.csv` are immutable; system predictions never approve records.
- Original JPEGs remain canonical; do not create enhanced training images.
- Do not touch quarantined data, `datasets/newdata`, training, ONNX, or Git.

## Files

- Create: `low_light_dataset/review_store.py`
- Create: `tests/test_review_store.py`

## Required interface

```python
class RevisionConflict(RuntimeError): ...
class CandidateChangedError(RuntimeError): ...

class ReviewStore:
    def __init__(self, work_root: Path, candidate_root: Path) -> None: ...
    def initialize(self, manifest_path: Path) -> dict[str, Any]: ...
    def summary(self) -> dict[str, Any]: ...
    def get(self, stem: str) -> ReviewRecord: ...
    def update(self, stem: str, patch: dict[str, Any], expected_revision: int,
               actor: str = "human") -> ReviewRecord: ...
```

State files are `<work_root>/annotation_state.json`, `.last-good`, `.tmp`, and `annotation_history.jsonl`.

## TDD requirements

1. Write `tests/test_review_store.py` before production code.
2. Run `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v` and record RED.
3. Write the minimal store, run focused GREEN, then run:
   `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store tests.test_review_models -v`.
4. Report both RED and GREEN evidence in `.superpowers/sdd/task-2-report.md`.

## Manifest and initialization contract

Read `candidate_root/manifest.csv` or the explicit `manifest_path`. Required columns are `stem` and `image_sha256`. Every stem must be unique; every `<candidate_root>/frames/<stem>.jpg` must exist; each current hash must equal the manifest hash. Refuse an empty manifest. Sort records by stem for deterministic state output.

`initialize` creates schema version 2 records with revision 0, status `unreviewed`, empty anchors/tags/warnings/notes, no exclusion reason or timestamps, `suggestion_modified=False`, `second_review_required=False`. It refuses to overwrite an existing valid state. If called again with the identical valid manifest/state, it returns the existing state idempotently. If the existing state covers different stems or hashes, raise a stable error.

## Atomic persistence contract

- Serialize UTF-8 JSON with sorted keys and indentation to `annotation_state.json.tmp`.
- Flush and `os.fsync` the temporary file.
- When a current valid state exists, copy it byte-for-byte to `annotation_state.json.last-good` and fsync the backup before replacing.
- Replace `.tmp` onto `annotation_state.json` atomically with `Path.replace`.
- Do not leave `.tmp` after success; a stale `.tmp` on startup is ignored and reported in summary.

## Load and recovery contract

If current state is absent, operations other than `initialize` raise `FileNotFoundError`. If current JSON is corrupt but `.last-good` is valid, load the backup without overwriting or deleting the corrupt current file and return `recovered_from_last_good=True` in `summary`. If both are corrupt or incompatible, raise a stable state error.

Every `get` and `update` recomputes the candidate JPEG SHA-256 and raises `CandidateChangedError(stem)` if it differs. Unknown stems raise `KeyError`.

## Update contract

- Compare `expected_revision` with the current record revision and raise `RevisionConflict` before any mutation if stale.
- Editable keys are exactly: `status`, `anchors`, `interference_tags`, `exclusion_reason`, `warnings`, `suggestion_modified`, `first_reviewed_at`, `second_review_required`, `second_reviewed_at`, `notes`.
- Reject keys outside this set, including `stem`, `image_sha256`, and `revision`.
- Merge the patch into the serialized record, increment revision exactly once, deserialize via `record_from_dict`, and reject any `validate_record` errors without writing files.
- Write the new complete state atomically. Only after state success, append one UTF-8 JSON line to history with UTC ISO time, actor, stem, prior revision, new revision, `before`, and `after`; flush and fsync history.
- If state writing fails, history must not change. If history append fails after state succeeds, raise a dedicated audit error and make `summary()` report `audit_incomplete=True`; never roll the already published state backward.

## Summary contract

Return `schema_version`, total count, counts for all five statuses, `recovered_from_last_good`, `stale_temporary_present`, and `audit_incomplete`. Counts must derive from loaded records, not cached mutable values.

## Required tests

Cover initialization/hash preservation, idempotence, duplicate/empty/missing/hash-mismatch manifest failures, deterministic order, stale revision, unknown/forbidden patch keys, validation rejection without writes, successful revision/history, state-write failure without history, history failure flag, candidate mutation on get/update, corrupt-current recovery, corrupt-both failure, stale temp reporting, and summary counts. Use temporary real JPEG files and real SHA-256 values.

## Completion report

Write `.superpowers/sdd/task-2-report.md` with implementation summary, exact files, RED/GREEN evidence, self-review, and concerns. Return only short status, one-line tests, concerns, and report path.
