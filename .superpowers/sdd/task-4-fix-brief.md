# Task 4 Review Fix Brief

Fix the independent-review findings test-first and do not touch later-task files.

## Important: manifest stem and path containment

1. Reject service initialization when any manifest stem is not a safe single filename component. At minimum reject empty strings, `.`/`..`, any `/` or `\\`, `Path(stem).name != stem`, absolute/rooted/drive semantics, control/NUL characters, and stems that do not match the project's explicit conservative identifier grammar (ASCII letters, digits, `_`, `-`; no extension).
2. Defense in depth: resolve every original-media path under `candidate_root/frames` and verify its parent/containment; resolve every cache/sidecar target under `<work_root>/enhanced_preview_cache` and verify containment before reads, mkdir, temp creation, or writes.
3. Refuse a pre-existing cache root or traversed parent that is a symlink/reparse-point-like link where Python can identify it. Never follow a cache-root symlink to write outside `work_root`. Tests should use a symlink when supported and skip only when the platform forbids creating it.
4. Add tests for malicious manifest stems including Windows backslash traversal and encoded request traversal, and assert no outside file is read/written.

## Minor: exact HTTP method contract

Werkzeug automatically adds HEAD to GET rules. Explicitly reject HEAD (and other uncontracted methods) with 405 for these routes; add a focused test for `HEAD /health` and `OPTIONS /health`.

## Verification

Run fresh:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service tests.test_preannotation tests.test_review_store tests.test_review_models -v`

Append RED/GREEN evidence and exact security behavior to `.superpowers/sdd/task-4-report.md`.
