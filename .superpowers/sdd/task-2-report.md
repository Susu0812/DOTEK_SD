# Task 2 Report: Atomic Review Store and Audit History

## Status

Implemented the Task 2 review persistence boundary with strict test-first development.

## Implementation summary

- Added schema-version-2 initialization from an explicit manifest with required-column,
  empty, duplicate-stem, missing-JPEG, and SHA-256 verification.
- Added deterministic stem ordering and idempotent initialization that refuses an
  existing valid state with different stems or hashes.
- Added immutable-record loading through Task 1's `record_from_dict`, and all update
  validation through Task 1's `validate_record`.
- Added optimistic revision checks, the exact editable-key allowlist, single revision
  increments, candidate re-hashing on every `get` and `update`, and stable exception
  types (`RevisionConflict`, `CandidateChangedError`, `ReviewStateError`, `AuditError`).
- Added atomic UTF-8 JSON state writes using `.tmp`, flush plus `os.fsync`, byte-for-byte
  `.last-good` backup of a valid current state, and `Path.replace` publication.
- Added corrupt-current recovery from `.last-good` without modifying the corrupt file,
  stale-temporary reporting, and stable failure when no compatible state is available.
- Added post-publication JSONL audit history with UTC ISO timestamps, complete before
  and after records, revisions, actor, and fsync. State-write failure cannot append
  history; history failure raises `AuditError` without rolling state back.
- Added summary values derived from loaded records for all five statuses. Audit
  incompleteness is detected by missing revision events and is also latched in the
  active store when an append operation raises.

## Exact files

- Created `low_light_dataset/review_store.py`
- Created `tests/test_review_store.py`
- Created `.superpowers/sdd/task-2-report.md`

No Task 1, candidate, dataset, training, ONNX, quarantine, or Git files were modified.

## TDD evidence

### RED

Tests were written before production code.

Command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v
```

Observed before `low_light_dataset/review_store.py` existed:

```text
ImportError: Failed to import test module: test_review_store
ModuleNotFoundError: No module named 'low_light_dataset.review_store'
Ran 1 test in 0.001s
FAILED (errors=1)
```

This was the expected RED: the required production module did not yet exist.

### Focused GREEN

Same command after the minimal implementation and after refining the fixture to use
an embedded, decodable 2x2 JPEG:

```text
Ran 17 tests in 1.020s
OK
```

### Combined GREEN

Required command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store tests.test_review_models -v
```

Observed final result:

```text
Ran 34 tests in 0.950s
OK
```

## Coverage and self-review

The 17 Task 2 tests cover:

- initialization defaults and exact manifest-hash preservation;
- initialization idempotence and different-state refusal;
- duplicate, empty, missing-column, missing-file, and hash-mismatch manifests;
- deterministic JSON record order and successful `.tmp` cleanup;
- operations before initialization, unknown stems, and stale revisions;
- forbidden patch keys and validation rejection with unchanged state/history;
- successful update, exactly one revision increment, complete audit event, backup,
  and temporary-file cleanup;
- state-write failure without history and audit-write failure with published state;
- audit-incomplete reporting both on the active store and after reopening when an
  event is missing;
- candidate mutation detection on both `get` and `update`;
- corrupt-current recovery without overwriting it, corrupt-both stable failure;
- stale temporary reporting and counts for all five statuses.

Self-review against the brief found no scope expansion. The implementation uses only
standard-library persistence primitives and Task 1's public model functions. Tests
use temporary on-disk, valid JPEG images and SHA-256 values computed from their real
bytes. All state counts are recomputed at summary time.

## Concerns

No known blockers. Audit completeness across store reopenings is inferred from the
presence of every history revision up to each loaded record's revision; events newer
than a recovered `.last-good` state are intentionally ignored for that comparison.

## Independent-review remediation (2026-07-17)

### Fix summary

The persistence review found four defects in the first implementation. All four were
reproduced with regression tests before production changes and are now fixed:

1. Update provenance is explicit and independent from the free-form actor label.
   `system` and `model` origins cannot publish an after record with a final status
   (`positive`, `hard_negative`, or `excluded`) or a non-null `first_reviewed_at`.
   Rejected attempts leave state and history byte-for-byte unchanged. Non-final
   model/system suggestions remain supported.
2. `update()` now rejects state loaded from `.last-good` before candidate checking or
   any persistence mutation. The corrupt current, backup, stale temporary, and audit
   history files remain unchanged.
3. Audit reopen validation now checks every JSONL event structurally and as a chain:
   all required fields; UTC-aware ISO time; non-empty actor; allowed origin; known
   stem; non-negative integer revisions; contiguous `prior_revision -> new_revision`;
   valid before/after records; matching stem/hash/revisions; equality between one
   event's `after` and the next event's `before`; and correspondence with loaded state.
   Malformed, missing, duplicated, out-of-order, incomplete, or inconsistent events
   set `audit_incomplete=True` instead of escaping as parser/type errors.
4. The full read/check/write/audit update sequence is serialized for a normalized
   `work_root`. A per-process `RLock` coordinates store instances in one process and
   `annotation_state.lock` adds a one-byte OS advisory lock (`msvcrt` on Windows,
   `fcntl` on POSIX) for separate processes. The deterministic interleaving test proves
   that one revision-0 writer succeeds and the loser receives `RevisionConflict`.

The minor atomic boundary request was also covered: a real mocked `Path.replace`
failure leaves current state and history unchanged, preserves the byte-for-byte
last-good backup, and leaves the unpublished `.tmp` for stale-temp reporting.

### Exact API change

`ReviewStore.update` is now:

```python
def update(
    self,
    stem: str,
    patch: dict[str, Any],
    expected_revision: int,
    actor: str = "human",
    origin: str = "human",
) -> ReviewRecord
```

- `origin` accepts exactly `human`, `system`, or `model`; the default preserves
  ergonomic existing human calls.
- `actor` remains a free-form label but must be a non-empty string.
- Every new audit event includes `origin` as a required field.
- Invalid actor/origin raises `ValueError`; a non-human approval/finalization attempt
  raises `PermissionError`.

### Review-fix RED evidence

The new regression tests were added before modifying production code.

Command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v
```

Clean reviewed-defect RED:

```text
Ran 29 tests in 1.930s
FAILED (failures=24, errors=12)
```

Key observed failures were: both concurrent revision-0 writers succeeded; recovered
state accepted an update; `origin` was not accepted or recorded; and malformed,
duplicated, out-of-order, or identity-inconsistent events were not reported as
incomplete. The missing `origin` API naturally produced `TypeError`/`KeyError` errors.

A self-review boundary test was also observed RED before its type guard:

```text
Ran 2 tests in 0.262s
FAILED (errors=2)
```

Both errors were the unhashable non-string `origin` case, one at the public API and
one during reopen audit validation.

### Final GREEN evidence

Focused Task 2 verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v
Ran 29 tests in 2.394s
OK
```

Required combined verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store tests.test_review_models -v
Ran 46 tests in 2.473s
OK
```

### Files changed for remediation

- `low_light_dataset/review_store.py`
- `tests/test_review_store.py`
- `.superpowers/sdd/task-2-report.md`

No later-task file was modified.

### Residual concerns

- Cross-instance thread interleaving is deterministically tested on Windows. The
  cross-process layer uses standard OS advisory locks but was not exercised by a
  multiprocessing test; advisory-lock guarantees can also depend on network
  filesystem behavior. The intended work root is local storage.
- Pre-remediation audit lines do not contain the newly required `origin` and will be
  reported as incomplete on reopen. This is intentional fail-closed behavior; no
  migration was requested.
- If history fsync raises after the complete line has already reached the visible file,
  the active store latches `audit_incomplete=True`; after a process restart the
  structurally complete line may appear valid because no separate durable audit-failure
  marker is part of the specified state-file contract.

## Second-review authorization closure (2026-07-17)

### Root cause and implementation

The first remediation checked only explicit patch keys before the record merge. A
non-human caller could therefore update notes on an already-final record: the merged
after record silently retained its final status and non-null first-review timestamp.
The same semantic authorization check was absent from audit-history reopen validation.

The fix replaces patch-key authorization with one shared after-record predicate used
by both paths:

- `human` may publish any after record accepted by Task 1 validation.
- `system` and `model` may publish only when the merged after status is non-final and
  `after.first_reviewed_at is None`.
- Live authorization runs after deserializing the merged after record but before any
  state/history write. A denial raises `PermissionError` with byte-for-byte zero side
  effects.
- Audit reopen applies the identical predicate to each validated event's after record;
  a violation makes `summary()["audit_incomplete"]` true.
- A non-human caller can move a previously final record back to an unapproved state
  only by explicitly setting a non-final status and clearing `first_reviewed_at`.

There is no signature change beyond the prior `origin` addition.

### Second-review RED evidence

The four closure regressions were written before the production predicate changed.

Command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store.UpdateTests.test_nonhuman_cannot_modify_an_existing_final_record tests.test_review_store.UpdateTests.test_nonhuman_can_explicitly_return_final_record_to_unapproved_state tests.test_review_store.AuditValidationTests.test_reopen_flags_nonhuman_event_with_final_or_first_reviewed_after tests.test_review_store.AuditValidationTests.test_reopen_flags_nonhuman_patch_to_previously_final_record -v
```

Observed RED:

```text
Ran 4 tests in 0.510s
FAILED (failures=5, errors=1)
```

Both non-human origins modified existing final records; structurally valid forged
history with final/first-reviewed after records remained accepted; and the old
patch-key rule incorrectly rejected an explicit transition back to unapproved state.

### Second-review final GREEN evidence

Focused Task 2 verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v
Ran 33 tests in 3.605s
OK
```

Combined verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store tests.test_review_models -v
Ran 50 tests in 3.030s
OK
```

The four new focused closure regressions also pass independently (4/4 in 0.455s).

### Additional residual concern

No new residual authorization concern is known within the specified fields. The
shared rule intentionally does not prohibit non-human second-review metadata when the
after record remains non-final and has no first-review timestamp; Task 1 record
validation remains the authority for those field combinations.
