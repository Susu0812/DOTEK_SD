# Task 4 Report: Loopback WSGI Service and Port Fallback

## Scope and files

Implemented Task 4 only. No persistent service was started, no Task 5 browser assets
were created, and candidates, checkpoints, training data, and Tasks 1-3 were not
modified.

Exact files changed:

- `low_light_dataset/review_service.py` (created)
- `tests/test_review_service.py` (created)
- `.superpowers/sdd/task-4-report.md` (created)

## TDD evidence

Initial RED command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v
```

Result: exit 1; the loader ran 1 failed test module and raised the expected
`ModuleNotFoundError: No module named 'low_light_dataset.review_service'`. The
production module did not exist at that point.

Initial GREEN used the same focused command. Result: 16 tests passed, 0 failures,
0 errors (`Ran 16 tests in 4.616s`).

During self-review, a second RED reproduced malformed unhashable JSON control/data
values being mapped to 500 instead of 400:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service.PatchTests.test_client_timestamps_malformed_controls_and_invalid_final_states_are_400 -v
```

Result before the fix: 1 test ran with 2 subtest failures (`action=[]` and
`status={}`, both actual 500 versus expected 400). After type-safe policy checks,
the same targeted command passed 1/1.

Fresh final focused verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v
```

Result: exit 0; 16 tests passed, 0 failures, 0 errors, no skips
(`Ran 16 tests in 3.319s`).

Fresh regression verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service tests.test_preannotation tests.test_review_store tests.test_review_models -v
```

Result: exit 0; 78 tests passed, 0 failures, 0 errors, no skips
(`Ran 78 tests in 7.409s`).

## WSGI route contract

`create_app(candidate_root, work_root, preannotation_path=None,
static_root=None)` initializes the existing `ReviewStore` against
`candidate_root/manifest.csv`; incompatible state propagates as an initialization
failure and is never replaced. The returned callable exposes `review_store` only
as an explicit diagnostic/test-access attribute.

Implemented exact routes:

- `GET /health` returns exactly `ok`, `candidate_count`, and `record_count`.
- `GET /api/summary` returns the live `ReviewStore.summary()` object.
- `GET /api/records` and `GET /api/records/<stem>` join chronological manifest
  metadata with human state; preannotations are nested under `preannotation` and
  cannot overwrite human fields. Detail includes `previous_stem` and `next_stem`.
- `PATCH /api/records/<stem>` is the only mutation route.
- `GET /api/review-queue` returns deterministic chronological `stems` and joined
  `records` for unreviewed or outstanding second-review work.
- Original and enhanced JPEG routes accept exact manifest stems and `.jpg` only.
- Static serving accepts direct child files only; nested paths, traversal, encoded
  traversal, backslash traversal, unknown stems, and unexpected suffixes are 404.
- Methods outside each contract are JSON 405 and do not mutate state.

## PATCH policy and error mapping

- JSON must be an object with a non-boolean, non-negative integer `revision`.
- `action` is exactly `draft`, `finalize`, or `second_review`; `actor` is ignored as
  a client authorization choice. Store calls always use
  `actor="local_web_reviewer", origin="human"`.
- Client `first_reviewed_at` and `second_reviewed_at` are rejected. Final status
  transitions receive a server UTC first-review time only if one is absent.
  Explicit second review receives a server UTC second-review time, preserves the
  first time, and creates both times when necessary.
- Draft editable fields are limited to status, anchors, interference tags,
  exclusion reason, warnings, and notes. A draft entering a final status is treated
  as finalize for timestamp purposes. Finalize/second-review actions require a
  valid final status.
- Malformed JSON/control values, forbidden timestamps, patch policy violations,
  and record validation are stable 400 JSON errors; unknown records/media are 404;
  revision conflicts and candidate changes are stable 409 errors; persistence,
  audit, and state failures are stable path-free 500 errors.
- Preannotation and client-supplied actor/origin values cannot authorize or approve
  a record.

## Enhanced preview cache

The cache is restricted to `<work_root>/enhanced_preview_cache/<stem>.jpg` plus a
same-stem JSON sidecar. Every request first calls the store's candidate/hash
verification. Reuse requires a decodable 640x480 JPEG, schema version 1, exact
manifest source SHA-256, a complete finite enhancement-parameter object, and
matching output format/dimensions/byte-size/SHA-256 metadata.

Missing, corrupt, dimension-invalid, or stale entries regenerate from the verified
canonical JPEG through `enhance_low_light` and `save_jpeg`. JPEG and JSON are
written to exclusively created randomized same-directory temporary files and then
replaced into place; only those owned temporary paths are cleaned. Tests assert
generation, reuse, all invalidation paths, no leftover owned temporary files, and
an unchanged candidate tree byte/path snapshot.

## Port fallback result

`find_available_port` accepts only the literal host `127.0.0.1`, validates inclusive
port bounds, probes in increasing order with a new temporary socket per port, and
closes every probe. A fully occupied range raises
`no_loopback_port_available_<start>_<end>`.

Actual final test result: the test successfully bound/listened on 127.0.0.1:8765,
confirmed 8766 was available, and `find_available_port()` returned **8766**. A
separate dynamic-port test rebound the returned port immediately, confirming the
probe socket had closed.

## Residual concerns

- Port discovery is necessarily a probe rather than a reservation; another process
  can claim the returned port before the future Task 8 CLI binds it. Task 8 should
  handle bind failure by probing again rather than exposing a non-loopback host.
- JPEG and sidecar publication consists of two atomic file replacements, not a
  cross-file transaction. A crash between replacements can leave a mismatched pair;
  strict sidecar/output validation makes the next request regenerate it safely.
- This task supplies no browser UI and intentionally does not launch or persist a
  server. Task 5 assets and Task 8 CLI integration remain future work.

## Independent-review security fixes

The Task 4 independent review identified missing Windows-safe manifest stem
validation, insufficient resolved path containment, cache link/reparse escape risk,
and Werkzeug's implicit HEAD support. These findings were reproduced and fixed
test-first without changing later-task files.

Clean targeted RED command:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service.InitializationSecurityTests tests.test_review_service.RouteTests.test_head_and_options_are_not_implicitly_added_to_get_routes -v
```

Before the fix, this ran 2 test methods with 16 assertion failures: all 15 malicious
stem subcases failed to produce the required pre-access
`ValueError("unsafe_manifest_stem")`, and `HEAD /health` returned 200 instead of
405. The `/` and Windows `\` traversal subcases demonstrably reached the mocked
candidate hash boundary before rejection. The corrected test fixture writes the
NUL-containing manifest directly, so RED failures came from service behavior rather
than CSV fixture setup.

Targeted GREEN used the same command: exit 0; 2/2 test methods passed
(`Ran 2 tests in 0.572s`).

Fresh post-fix focused verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v
```

Result: exit 0; 21 tests ran, 18 passed and 3 were conditionally skipped, with 0
failures and 0 errors (`Ran 21 tests in 3.972s`). The skips are the three real
file/directory symlink escape tests: this Windows account rejected symlink creation
with WinError 1314. The tests execute on platforms/accounts that permit symlinks and
skip only for that platform denial.

Fresh post-fix regression verification:

```text
D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service tests.test_preannotation tests.test_review_store tests.test_review_models -v
```

Result: exit 0; 83 tests ran, 80 passed and the same 3 symlink tests were skipped,
with 0 failures and 0 errors (`Ran 83 tests in 8.692s`).

Exact added security behavior:

- The service reads and validates manifest stems before initializing `ReviewStore`.
  Stems must match ASCII `[A-Za-z0-9_-]+` and are additionally rejected for empty,
  dot/dot-dot, slash/backslash, non-basename, absolute/rooted/drive, extension,
  control/NUL, whitespace, Unicode, or other nonconservative forms. Unsafe stems
  cannot reach candidate hashing or create work state.
- Every canonical original is a resolved direct child of the resolved
  `candidate_root/frames`; a linked/reparse frames root or candidate child and any
  resolved parent escape are rejected as `candidate_changed` before candidate hash
  or media reads. Containment is checked again after Store verification.
- Work/cache roots and both preview targets are checked with `lstat` for symlink or
  Windows reparse attributes and with resolved direct-parent equality. Checks occur
  before cache reads, directory creation, temporary creation, replacement, and the
  final response read. Unsafe cache paths map to path-free `save_failed` and cannot
  read or write the external sentinel trees used by the tests.
- Encoded Windows backslash traversal is explicitly covered and remains 404.
- After route matching, implicit Werkzeug HEAD is explicitly rejected with 405;
  OPTIONS and every other uncontracted method remain 405. HEAD correctly carries no
  response body under HTTP semantics.
