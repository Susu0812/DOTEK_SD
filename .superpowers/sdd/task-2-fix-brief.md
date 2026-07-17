# Task 2 Fix Brief

Fix the reviewed persistence defects in `low_light_dataset/review_store.py` and extend `tests/test_review_store.py` with regression tests. Work test-first and do not touch later-task files.

## Required fixes

1. Enforce at the persistence boundary that a system/model-originated update cannot approve/finalize a record and cannot set `first_reviewed_at`. Use an explicit, unambiguous origin value separate from the free-form actor label. Preserve ergonomic human updates and record the origin in audit events. Add negative tests proving system-originated attempts have no state/history side effects.
2. If `_load_state()` recovered from `annotation_state.json.last-good`, reject `update()` without modifying the corrupt current, backup, temporary, or history files. Add a regression test.
3. Validate audit history structurally and as a revision chain on reopen: required event fields, valid time/actor/origin, matching stem, contiguous `prior_revision -> new_revision`, and before/after identity plus revision correspondence. Malformed, incomplete, duplicate, out-of-order, or inconsistent events must make `summary()["audit_incomplete"]` true. Add focused regression tests.
4. Serialize the whole update read/check/write/audit sequence across separate `ReviewStore` instances for the same work root so two concurrent revision-0 writers cannot both succeed. The loser must receive `RevisionConflict`; no lost update and no shared-temp race. Add a deterministic concurrency test.
5. If practical while touching persistence, add one lower-level atomic-write failure test covering a real failure boundary. Treat this as minor; do not broaden scope.

## Verification and handoff

Run:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store -v`

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_store tests.test_review_models -v`

Update `.superpowers/sdd/task-2-report.md` with RED/GREEN evidence, exact API change, test counts, and any residual concern. Do not claim completion if any required fix is untested.

## Second-review authorization closure

The first remediation still allowed a non-human origin to patch an already-final record while preserving its final approval. Fix this test-first:

1. Authorization must be evaluated against the merged `after` record, not only explicit patch keys. `system` or `model` may never publish an `after` record in `positive`, `hard_negative`, or `excluded`, and may never publish a non-null `first_reviewed_at`. A non-human suggestion must explicitly remain or return to a non-final review state.
2. Apply the same authorization predicate while validating history on reopen. A structurally valid event with a non-human origin and final/first-reviewed `after` must make `audit_incomplete=True`, including a patch to a previously final record.
3. Add regression tests for both the live update path and forged-but-structurally-valid history path, proving fail-closed behavior and zero live-update side effects.
4. Re-run both required GREEN commands and append the second-review RED/GREEN evidence to the Task 2 report.
