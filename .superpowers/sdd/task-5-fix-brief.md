# Task 5 Review Fix Brief

Fix all independent-review Critical and Important findings test-first. Modify only Task 5 files and report.

## 1. Never auto-finalize after a failed pending save

- `confirmClassification` must inspect the result of `awaitPendingAutosave()`.
- If it is false for 409, validation, network, or save failure, stop immediately: do not issue the final PATCH, do not update the classification preview as saved, and require a new explicit user click/confirmation.
- The 409 handler may update the local expected revision and must preserve local fields, but cannot silently retry any save.
- Add an observable/static contract test, and structure the code so browser smoke can count PATCHes.

## 2. Preserve edits made while a save is in flight

- Add a monotonically increasing local edit generation/dirty version, incremented for anchor add/move/delete/undo/redo/clear and text/tag/reason changes.
- `performSave` captures the generation and a draft snapshot when creating its payload.
- On success, always advance the server revision and safe server metadata, but only replace local editable anchors/fields with returned values if the local generation still equals the captured generation.
- If newer edits exist, preserve them visibly, keep/schedule the next serialized autosave with the newly returned revision, and report that newer edits remain pending. Never let an older response overwrite newer geometry.
- Avoid infinite autosave loops. Final/hard-negative responses with newer incompatible local edits must fail safe and require explicit reclassification rather than silently discarding geometry.

## 3. Exclusion reason draft compatibility and class transitions

- Never send a non-null `exclusion_reason` with a non-`excluded` status.
- Finalizing positive, hard negative, or needs-second-review explicitly sends/derives `exclusion_reason: null` so a prior excluded record can transition cleanly.
- Finalizing excluded sends the selected allowed reason.
- Because the Task 1 record schema intentionally forbids storing a reason on non-excluded records, preserve an uncommitted selected reason as browser-local draft state keyed by stem (sessionStorage with guarded access is acceptable) while backend autosave persists all compatible fields. Restore it on returning/reload; remove it after a successful non-excluded classification and retain/sync it after excluded classification. Document this deliberate split.
- Add tests/contracts for unreviewed reason edit, excluded finalize, and excluded-to-other-class clearing.

## 4. Correct queue/event filtering and selection consistency

- Event filter must use candidate manifest `source` and/or `event_reason`, not preannotation/revision heuristics. Provide understandable `base` and `event` options at minimum.
- Use `/api/review-queue` as the deterministic work queue for unreviewed/needs-second-review views; allow explicit status filters to inspect final records from the chronological record list.
- After filters remove the current record, load the selected first matching record (after pending-save handling), rather than changing only the `<select>` value. Empty results must not leave a misleading selected option.
- Keep record ordering deterministic and explain queue-vs-chronological behavior in the report.

## 5. Last-navigation intent wins

- Add a monotonic navigation request token or `AbortController`. After every awaited step, stale navigation calls must return without changing detail, canvas, select, thumbnails, or status text.
- Rapid previous/next/select actions must leave the final requested record loaded.

## 6. Preview toggle scope

- Toggle only the canvas background between original/enhanced. Current and adjacent thumbnails remain canonical originals.

## Verification

- Extend `tests/test_review_service.py` with focused contract checks for every fix. Static assertions alone are insufficient if a simple server-side or Python-level deterministic behavior can be tested, but do not add new dependencies.
- Run fresh focused:
  `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`
- Run fresh full:
  `D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest discover -s tests -v`
- Update `.superpowers/sdd/task-5-report.md` with fix RED/GREEN evidence and residual concerns.
- Do not claim browser smoke; root performs it after approval.

## Second closure: sub-event generation and filter invalidation

Independent re-review found three remaining races. Fix them test-first:

1. Mark the local edit generation before or at every `pointermove` mutation that changes anchor geometry, not only on `pointerup`. A PATCH response arriving between move and up must see a newer generation and must not restore its old anchors. Keep one undo snapshot per drag and avoid unnecessary autosave spam; the generation guard and autosave scheduling may use different moments.
2. Remove the unconditional old-draft restore in the 409 remote-detail GET path. The GET does not need to alter editable state; update only revision/safe metadata. If any restoration remains, it must be conditional on an unchanged generation and must never overwrite edits made while the GET was in flight.
3. Every filter application is a navigation intent. Invalidate the current navigation token before calculating/handling visible results, including current-still-visible and empty-result branches. Any older detail response must be unable to update the UI after a filter change.
4. Add focused contract/regression coverage for these exact guards, then run fresh focused and full discovery again and append evidence to the report.
