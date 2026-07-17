# Task 7 implementation and test report

## Implementation

- Added `low_light_dataset/review_export.py` with deterministic, monotonic
  second-review assignment through audited `ReviewStore.update` calls.
- Added read-only, fail-closed review and staging-bundle validation with stable
  error codes for coverage, review state, candidate integrity, masks, pairing,
  formal-dataset collisions/leakage, audit completeness, stale state, and unsafe
  paths.
- Added transactional export through a sibling `.staging` tree, canonical JPEG
  byte copies, rasterized positive/all-zero hard-negative labels, overlays,
  deterministic contact sheets, atomic JSON reports, staging revalidation,
  formal-dataset before/after snapshots, and final atomic rename.
- Export performs read-only review validation and a read-only assignment plan
  first. If explicit second-review assignment is still required it raises
  `second_review_assignment_required` without changing state or creating staging.
- Output symlink/reparse ancestors are rejected and rechecked immediately before
  publication. Overlays are generated from the verified staging image copy.
- Staging validation decodes and deterministically recomputes overlay/contact
  JPEGs and cross-checks report schemas, counts, stems, and artifact hashes.
- Every qualifying audit event is honored, including assignment-actor events;
  the audit sample remains idempotent by ranking the stable set of intrinsically
  plain positives independently from mandatory audit-history reasons.
- The implementation neither marks candidates reviewed nor merges or trains.
  Only explicit second-review requirements are assigned, with human origin and a
  non-empty assignment actor as required.

## TDD evidence

The focused test file was written before production code. The initial focused
run executed 12 tests and failed all 12 with the expected
`low_light_dataset.review_export is not implemented` assertion. After the
minimal implementation and one specification-alignment correction to the audit
history expectation, the focused suite passed.

An independent safety review then identified five gaps. Seven new behavioral
regressions were added first; they failed against the initial implementation as
expected (artifact validation, output ancestry/recheck, staging-copy provenance,
and non-mutating explicit assignment), and an isolated history regression also
failed as expected. The implementation was then corrected until all 19 focused
tests passed.

A final delta review added two more fail-closed regressions. Report assignment,
formal snapshot, and canonical validation-field tampering produced three RED
failures. A simulated report-hash read race produced an uncaught `PermissionError`
in RED. Exact report provenance comparison and per-file `OSError` handling were
then implemented; unreadable report artifacts now return the stable structured
code `bundle_report_unreadable`.

Production read-only validation then exposed legitimate formal dataset stems
containing decimal dots (for example `video1_10.0x_frame_00000`). A regression
first failed with `formal_stem_unsafe`; formal inventory validation now permits
safe `[A-Za-z0-9_.-]+` stems while still rejecting empty/dot-only names. Candidate
manifest stems remain governed by the original strict no-dot `SAFE_STEM` rule.

## Verification

- Focused: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_export -v`
  - Final: `Ran 22 tests in 24.996s` / `OK`
- Full: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v`
  - `Ran 172 tests in 63.421s` / `OK (skipped=3)`
  - This full run followed the five main safety-review fixes. The two final,
    localized report-validation refinements were reverified by the complete
    21-test focused suite as requested.
  - The three skips are pre-existing Windows privilege-dependent symlink tests.
- Syntax: `D:\Anaconda3\envs\hosebot_cv\python.exe -m py_compile low_light_dataset\review_export.py tests\test_review_export.py`
  - exit code 0

All export tests use temporary fixtures. No production annotation state, formal
dataset, or production output location was mutated.
