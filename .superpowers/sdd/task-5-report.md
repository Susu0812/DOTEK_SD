# Task 5 Report: Browser Annotation UI

## Outcome

Implemented Task 5 only: a packaged browser annotation workbench, safe `/` and static asset serving, and server-derived `suggestion_modified` metadata. No persistent server or browser smoke was started; the root agent owns the independent in-app-browser smoke.

## TDD evidence

### RED

Command:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Initial result before production changes:

- `Ran 29 tests in 6.874s`
- `FAILED (failures=29, skipped=3)`
- Expected failures showed `/` returning 404, packaged JS/CSS returning 404, missing UI contract fragments, HEAD/OPTIONS on `/` returning 404 instead of 405, and changed suggestion geometry leaving `suggestion_modified` false.

### GREEN

Final focused run after the last UI error-path change:

- `Ran 29 tests in 6.087s`
- `OK (skipped=3)`
- 26 passed, 0 failures, 0 errors, 3 skipped because the Windows account cannot create test symlinks/reparse points.

Latest full regression run:

- `Ran 145 tests in 44.926s`
- `OK (skipped=3)`
- 142 passed, 0 failures, 0 errors, 3 skipped for the same Windows symlink privilege limitation.
- This full run preceded only the final client-side change that aborts record navigation when a pending save fails; the focused suite was rerun after that change and passed.

An earlier full run also passed `145` tests with `3` skips in `39.510s`.

## Independent-review fix cycle

### Fix RED

After the independent review, four focused contract tests were added for the six requested fixes.

- Focused command: `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`
- Initial fix result: `Ran 33 tests in 8.603s`, `FAILED (failures=21, skipped=3)`.
- The failures demonstrated that pending-save results were ignored before finalize, in-flight responses had no edit-generation guard, reasons were not status-compatible or browser-local, filters used preannotation/revision heuristics instead of manifest event data, queue ordering/selection was not used, navigation lacked a last-intent token, and preview toggling also changed the current thumbnail.
- A follow-up targeted RED exposed a remaining silent retry path: a queued `.then()` could run another autosave after a 409. The targeted test ran once and failed once before the failure latch was added.
- A second targeted RED demonstrated that an empty work queue still initialized a hidden chronological final record.

### Fix GREEN

- Final focused: `Ran 33 tests in 6.730s`, `OK (skipped=3)` — 30 passed, 0 failures/errors, 3 privilege skips.
- Final full regression: `Ran 149 tests in 46.703s`, `OK (skipped=3)` — 146 passed, 0 failures/errors, 3 privilege skips.

### Fix behavior delivered

- Pending save failure now blocks classification immediately. A 409 updates the expected revision and preserves local fields, but queued autosaves stop behind a failure latch; only a later explicit Save or newly confirmed classification can retry.
- Every editable change advances a monotonic generation. Saves capture both generation and draft snapshot. Responses always advance safe server metadata/revision, but editable fields are applied only when the captured generation remains current. Newer draft edits stay visible and continue through the already serialized autosave queue; newer edits during a final response require explicit reclassification.
- Exclusion reasons are stored per stem in guarded `sessionStorage` with an in-memory fallback. Non-excluded payloads always send `exclusion_reason: null`; excluded finalization sends the allowed selected reason. Successful transitions to positive, hard negative, or needs-second-review clear the local reason.
- Event filtering now uses manifest `source`/`event_reason` with understandable base/event choices. All/unreviewed/needs-second-review work views follow `/api/review-queue` ordering; explicit final-status inspection follows the chronological record list.
- When filters remove the current record, the first matching record is actually loaded after pending-save handling. Empty filters leave no selected option, and empty initial queues do not fall back to hidden final records.
- Navigation uses a monotonic request token and checks it after every awaited save/fetch boundary, so stale responses cannot change detail, canvas, select, thumbnails, or status.
- Preview toggle changes only the canvas background. Current, previous, and next thumbnails always use canonical originals.

## Second closure: sub-event generation and filter invalidation

### Closure RED

Three targeted guard tests were added before changing production JavaScript.

- The first targeted run executed 3 tests: pointermove generation and 409 draft preservation failed as expected; the filter test initially raised a test-harness `ValueError` because it indexed the deliberately missing guard.
- The filter test was corrected to assert guard presence before positional checks, without changing production code, and reran as an expected assertion failure.
- The clean RED evidence therefore showed all three missing behaviors: no generation mark inside pointermove, unconditional `restoreDraft` after the 409 detail GET, and no navigation-token invalidation at filter entry.

### Closure GREEN

- Targeted closure tests: `Ran 3 tests in 0.788s`, `OK`.
- Fresh focused service/UI suite: `Ran 36 tests in 8.353s`, `OK (skipped=3)` — 33 passed, 0 failures/errors.
- Fresh full regression: `Ran 152 tests in 42.378s`, `OK (skipped=3)` — 149 passed, 0 failures/errors.

### Closure guards delivered

- Every pointermove that produces different normalized geometry increments `editGeneration` immediately before rendering. Repeated/no-op moves do not increment it. Undo history and debounced autosave remain one action at pointerup, avoiding autosave spam.
- The 409 remote detail GET no longer captures or restores editable draft state. It merges only safe non-editable metadata when the same stem is still active, so edits made while the GET is in flight cannot be overwritten.
- `applyFilters` increments `navigationToken` before reading filters or calculating results. This invalidates older detail requests even when the current record remains visible or the result is empty; branches that load another record continue through the normal last-intent token checks.

## Files changed

- `low_light_dataset/review_web/index.html` (created)
- `low_light_dataset/review_web/app.js` (created)
- `low_light_dataset/review_web/styles.css` (created)
- `low_light_dataset/review_service.py`
- `tests/test_review_service.py`
- `.superpowers/sdd/task-5-report.md` (created)

No Task 1-4 model, store, candidate, checkpoint, training, LabelMe, export, or CLI file was changed.

## UI contract delivered

- UTF-8 page with the exact permanent safety banner, all required stable IDs, accessible labels/buttons, and a 640×480 intrinsic canvas.
- Original image is the default; enhanced preview changes only the image URL. Suggestions render amber and the human draft renders red without replacing human geometry.
- Pointer conversion uses the canvas bounding rectangle, coordinate clamping, 6-display-pixel selection, integer y, finite x, deterministic y deduplication, and monotone-y rendering.
- Bounded undo/redo, selected-anchor Delete/Backspace only in editor focus, confirm-before-clear, and no one-key class approval.
- Summary/records/detail/neighbors/queue loading; deterministic status/event/warning filtering; previous/current/next timestamps and thumbnails with disabled sequence ends.
- 500 ms draft autosave, serialized save attempts, navigation waiting for pending saves, and navigation cancellation on save failure.
- PATCH payload is allowlisted and omits client timestamps, origin, actor, image hash, and preannotation fields.
- Positive/hard-negative/excluded/needs-second-review validation and explicit class-naming confirmation dialogs; hard negative saves empty anchors only after confirmation.
- 409 refreshes the remote revision while restoring local anchors and fields; other failures retain local state and display a Chinese error.
- Responsive desktop multi-column styling keeps a 4:3 canvas visible and stacks below 760 px without horizontal overflow.

## Service contract delivered

- `GET /` serves the exact `index.html`; omitted `static_root` uses packaged `low_light_dataset/review_web`, while explicit roots remain supported.
- HTML/JS/CSS MIME types are deterministic. Index/static serving reuses direct-child, containment, and link protections; missing assets return stable 404 and read failures are caught as stable 500 responses without path leakage.
- HEAD/OPTIONS remain 405 on `/` and static routes.
- Preannotation remains nested and non-authoritative.
- Anchor PATCHes compare only serialized y/x geometry against preannotation, derive `suggestion_modified` server-side, preserve true after divergence, and reject client injection. Suggestions never create status or timestamps.

## Residual concerns / handoff

- `node --check low_light_dataset/review_web/app.js` could not run because `node` is not installed/on PATH (`CommandNotFoundException`). JavaScript contract assertions pass, but browser parsing/runtime behavior still needs the root agent's real in-app-browser smoke.
- Three link-escape tests were skipped because this Windows account lacks symlink creation privilege; the non-link traversal/containment tests passed.
- Browser-local reason persistence deliberately splits uncommitted reason state from the Task 1 server schema: backend autosave persists compatible fields while the per-stem reason remains in session storage until an excluded classification or cross-class cleanup.
- No persistent service and no manual browser smoke were run by this task agent.

## Browser-smoke remediation: banner integrity and summary counts

The root agent's real browser smoke exposed two UI integration issues. The packaged banner had suffered an encoding-damaged edit, and the initial summary rendered three unreviewed records as zero because the browser read legacy `status_counts`/`by_status` fields while `ReviewStore.summary()` publishes the authoritative mapping as `counts`.

The root-owned banner correction was preserved exactly as `增强图仅辅助观察，导出始终使用原图`. Its service/UI contract test also preserves the exact string and rejects U+FFFD replacement characters and private-use characters. This task agent did not alter that correction.

### Remediation RED/GREEN

- Added a service/UI contract test that first proves `/api/summary` returns `total: 3`, `counts.unreviewed: 3`, and `counts.needs_second_review: 0`, then requires the packaged UI to consume `summary.counts` first.
- Targeted RED: `Ran 1 test in 0.753s`, `FAILED (failures=1)`. The API assertions passed; the only failure was the missing `summary.counts` UI mapping.
- Minimal fix: `summary.counts || summary.status_counts || summary.by_status || {}`. `counts` is authoritative, with both legacy aliases retained as fallbacks.
- Targeted GREEN: `Ran 1 test in 1.791s`, `OK`.
- Fresh focused service/UI suite: `Ran 37 tests in 11.644s`, `OK (skipped=4)` — 33 passed, 0 failures/errors. Three skips are Windows symlink-privilege limitations; one is the environment-dependent port test because external process occupancy made port 8765 unavailable.
- Fresh full regression: `Ran 153 tests in 81.287s`, `OK (skipped=4)` — 149 passed, 0 failures/errors, with the same four skips.

The root agent retains ownership of the real browser smoke. This task agent made no browser or persistent-service operation during this remediation.
