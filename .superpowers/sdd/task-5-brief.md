# Task 5 Brief: Browser Annotation UI

Implement only Task 5 of the approved annotation-workbench plan, test-first. Create the three assets under `low_light_dataset/review_web/`, extend `tests/test_review_service.py`, and make only the minimal `low_light_dataset/review_service.py` changes needed to serve `/` and support safe draft metadata. Do not change Tasks 1-4 semantics, candidates, checkpoints, training, LabelMe, export, or CLI files.

Before implementation, demonstrate a focused RED because `/`/assets/required controls are missing. After implementation run focused and full current regression. The root agent will perform the real in-app-browser smoke after code review.

## Files

- Create `low_light_dataset/review_web/index.html`
- Create `low_light_dataset/review_web/app.js`
- Create `low_light_dataset/review_web/styles.css`
- Modify `low_light_dataset/review_service.py` minimally
- Extend `tests/test_review_service.py`
- Create `.superpowers/sdd/task-5-report.md`

## Service/static integration

- `GET /` serves the exact `index.html` from the review web root.
- When `static_root` is omitted, default to the packaged `low_light_dataset/review_web` directory. An explicit `static_root` remains supported for tests/integration.
- `/static/app.js` and `/static/styles.css` retain Task 4's direct-child/link/containment protections and correct MIME types.
- HEAD/OPTIONS remain 405 according to Task 4's exact method contract.
- If the default assets are missing/unreadable, return a stable 404/500 without path leakage.
- Preannotation remains nested and non-authoritative.
- If anchors are patched and a preannotation exists, derive `suggestion_modified` server-side by comparing serialized y/x geometry (small float tolerance is acceptable); do not trust a client-supplied `suggestion_modified`. Preserve true once human geometry has diverged, unless a clear server-side exact reversion policy is tested and documented. This flag never approves a record.

## Required DOM contract

`index.html` must be UTF-8, load `/static/styles.css` and deferred `/static/app.js`, and expose at minimum these exact IDs:

- `hose-canvas`, `preview-toggle`, `previous-frame`, `next-frame`
- `mark-positive`, `mark-hard-negative`, `mark-excluded`, `mark-needs-review`
- `undo`, `redo`, `clear-anchors`, `save-status`
- `status-filter`, `event-filter`, `warning-filter`, `record-select`
- `interference-tags`, `exclusion-reason`, `notes`, `warnings`
- `current-stem`, `current-timestamp`, `previous-timestamp`, `next-timestamp`
- `revision-indicator`, `save-result`, `summary-counts`, `classification-preview`
- previous/current/next visual elements or thumbnails with stable IDs.

Display a permanent visible banner containing exactly:

`增强图仅辅助观察，导出始终使用原图`

Do not include a one-key or keyboard shortcut that final-approves any class.

## Canvas and editing behavior

- Canvas intrinsic dimensions are exactly `width="640" height="480"`.
- Load original background by default from `/media/original/<encoded stem>.jpg`; preview toggle changes only to `/media/enhanced/...`, never anchor coordinates.
- Draw model/preannotation suggestion anchors/line in amber and current human draft anchors/line in red, clearly distinguishable.
- Convert pointer coordinates using the approved scale formula based on `getBoundingClientRect`, clamped to x `[0,639]`, y `[0,479]`.
- Pointer down within six display pixels of an anchor selects it for drag; otherwise add an anchor. Drag updates clamped coordinates. Store integer y and finite numeric x, keep anchors strictly sorted by y, and resolve duplicate y deterministically to one anchor.
- Delete/Backspace removes the selected anchor only while the canvas/editor has focus and not while typing in inputs.
- Ctrl/Cmd+Z and Ctrl/Cmd+Y implement bounded undo/redo; toolbar buttons do the same. Clear requires `window.confirm`.
- Render the final current centerline as monotone y segments through current anchors; never silently replace it with suggestions.

## Classification and saving

- Load summary, chronological records, detail, neighbors, and queue from Task 4 routes. Filters must support status/event/warning and update the selectable queue deterministically.
- Navigating previous/next or selecting another record waits for any pending draft autosave attempt before changing records.
- Debounce draft anchor/notes/tags/reason autosave by 500 ms. PATCH with the current revision and `action="draft"`; never send either timestamp, origin, actor authorization, image hash, or preannotation fields.
- Positive requires at least 3 anchors.
- Hard negative requires at least one allowed interference tag and, after explicit confirmation, saves empty anchors.
- Excluded requires an exclusion reason.
- Needs second review accepts notes/warnings and does not create a final timestamp.
- Every final classification button opens a confirmation dialog that explicitly names the chosen class before PATCH. No batch approval or single-key final approval.
- `save-status` explicitly saves the current draft/non-final state and reports success/failure.
- On HTTP 409, fetch the remote record to obtain the new revision and display a revision-conflict message while preserving local unsaved anchors/fields. Do not silently overwrite local edits.
- On other errors, keep local edits and show a clear Chinese error in `save-result`.

## Adjacent frames and visibility

- Show previous/current/next timestamps and images from chronological neighbors supplied by the service.
- Missing previous/next at sequence ends is clearly disabled/empty.
- Show warnings, preannotation confidence/source context, current revision, save status, and classification being confirmed.
- CSS must keep the 4:3 canvas materially visible at 1280x720, use a responsive multi-column desktop layout, and stack controls below 760 px width. Avoid horizontal overflow at narrow width.
- Use accessible labels, button types, visible keyboard focus, and sufficient color contrast.

## Required tests

Extend the service test suite with at least:

1. Initial RED: `/` missing or required controls absent.
2. `/` returns HTML and all required IDs/banner/canvas dimensions; asset URLs resolve with expected MIME types.
3. Default packaged static root works; explicit static root remains safe.
4. JavaScript contract includes clamped coordinate conversion, 6-pixel selection, y sorting/deduplication, undo/redo, confirm-before-clear/final classes, 500 ms debounce, timestamp-field omission, 409 preservation path, original/enhanced URL toggle, previous/next handling, and no one-key final approval binding.
5. CSS includes desktop 4:3 layout and `@media (max-width: 760px)` stacking.
6. Server derives/preserves `suggestion_modified` from human anchor changes and rejects client injection of that field; no status/timestamp is auto-created by suggestions.
7. HEAD/OPTIONS on `/` and static remain 405; traversal/link protections remain intact.

Run focused:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Run regression:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v`

## Handoff

Create `.superpowers/sdd/task-5-report.md` with RED/GREEN evidence and counts, exact files changed, UI/service contract, and residual concerns. Do not start a persistent server or claim manual browser smoke; the root agent owns that smoke after independent code review.
