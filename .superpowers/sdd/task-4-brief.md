# Task 4 Brief: Loopback WSGI Service and Port Fallback

Implement only Task 4 of the approved annotation-workbench plan with strict TDD. Create `tests/test_review_service.py`, demonstrate RED, then create `low_light_dataset/review_service.py`. Do not create Task 5 web assets, do not run a persistent server, and do not modify candidates, checkpoints, training data, or Tasks 1-3.

## Runtime and dependencies

- Use the installed Werkzeug 3.0.6 and Python standard library; do not add dependencies.
- Consume `ReviewStore`, `record_to_dict`, `enhance_low_light`, `save_jpeg`, candidate `manifest.csv`, optional `preannotation.json`, and later static files from `low_light_dataset/review_web`.
- The WSGI app must be directly testable with `werkzeug.test.Client` and `werkzeug.wrappers.Response`.

## Required public interfaces

Provide stable, documented signatures suitable for Task 8 CLI integration:

```python
def create_app(
    candidate_root: Path,
    work_root: Path,
    preannotation_path: Path | None = None,
    static_root: Path | None = None,
) -> WSGIApplication:
    ...

def find_available_port(
    host: str = "127.0.0.1",
    start: int = 8765,
    end: int = 8799,
) -> int:
    ...
```

`create_app` must initialize/validate the `ReviewStore` from `candidate_root/manifest.csv`; it must not silently replace incompatible state. It may attach explicit documented test-access attributes if useful.

## Exact loopback and port rules

- Only the literal host `127.0.0.1` is accepted. Reject `0.0.0.0`, `localhost`, IPv6 loopback, or any other text.
- Probe TCP ports in inclusive numeric order, using a temporary socket bound to `127.0.0.1` for each.
- Return the first available port.
- If none in the range is available, raise `RuntimeError("no_loopback_port_available_8765_8799")` for the default range (construct the same stable pattern from non-default bounds).
- Validate bounds and do not leave listening sockets open.

## Exact routes

Implement:

- `GET /health`: exactly `{"ok": true, "candidate_count": N, "record_count": N}` when usable.
- `GET /api/summary`: current `ReviewStore.summary()` JSON.
- `GET /api/records`: chronological manifest records joined with current review records and optional preannotation suggestion/metrics. Do not let preannotation fields overwrite human fields.
- `GET /api/records/<stem>`: one joined record plus chronological previous/next stems.
- `PATCH /api/records/<stem>`: optimistic human update described below.
- `GET /api/review-queue`: deterministic records/stems needing work, at minimum `unreviewed` and `needs_second_review`, in chronological order.
- `GET /media/original/<stem>.jpg`: verified canonical candidate JPEG only.
- `GET /media/enhanced/<stem>.jpg`: preview cache described below.
- `GET /static/<name>`: direct child files only from `static_root`; no nested paths/traversal. Task 5 will populate assets.

Only exact manifest stems are valid. Reject traversal, encoded traversal, unknown stems, unexpected suffixes, and static names for which `Path.name != requested_name`. Methods outside the contract return JSON 405 or normal Werkzeug 405, never mutate state.

## PATCH contract and server-owned timestamps

Request JSON must be an object containing integer `revision` (bool is invalid). Remaining editable fields pass to `ReviewStore.update(..., actor="local_web_reviewer", origin="human")` after service policy.

- Client JSON may never contain `first_reviewed_at` or `second_reviewed_at`; reject with HTTP 400 and stable error code. `revision`, `action`, and optional `actor` are control fields, not stored patch fields. Do not trust a client actor to choose authorization origin.
- `action` defaults to `draft`; allow exactly `draft`, `finalize`, and `second_review`.
- An ordinary draft may edit anchors/tags/reasons/warnings/notes/status, but may not create/overwrite timestamps. If its status enters `positive`, `hard_negative`, or `excluded`, treat it as finalize for timestamp purposes.
- When a human update changes/sets a final status and the stored `first_reviewed_at` is null, set it server-side to current UTC ISO-8601. Never overwrite an existing first-review time.
- `action="second_review"` is explicit. It may only publish a valid final status, must set `second_reviewed_at` server-side, and must not accept a client timestamp. Preserve the original first-review timestamp; if it is unexpectedly null, set first and second server-side in the same human event.
- `action="finalize"` must publish a final status. Draft autosaves and system suggestions cannot set either timestamp.

Map errors deterministically:

- malformed JSON/control fields/client timestamp/record validation: 400 JSON with stable `error` code;
- unknown stem/media: 404;
- `RevisionConflict`: 409 `revision_conflict`;
- candidate hash change: 409 `candidate_changed`;
- state/audit/write failures: 500 with a stable code (`save_failed` or more specific) and no traceback/body path leak.

## Enhanced preview cache

- Cache only under `<work_root>/enhanced_preview_cache/<stem>.jpg` with JSON sidecar `<stem>.json`.
- Sidecar must contain schema version, exact source SHA-256, enhancement parameters, and enough output metadata to validate the cache.
- Every request first verifies the candidate through the store/hash. Reuse only a decodable 640x480 cached JPEG whose sidecar source hash matches the current manifest hash and whose schema/fields are valid; otherwise regenerate from the canonical original using `enhance_low_light`.
- Publish preview and sidecar atomically using exclusively created randomized same-directory temporary files; never use a predictable caller-controlled temp path. Clean only owned temps.
- This cache is preview-only and must never be exposed as canonical/original data or written under the candidate root.

## Required tests

Use a temporary three-candidate fixture with real decodable 640x480 JPEGs and manifest hashes. Patch enhancement for speed/determinism. Cover at minimum:

1. Initial import RED.
2. `/health` exact count JSON; summary, chronological list/detail, neighbor fields, queue, optional suggestions merged under a non-authoritative namespace.
3. Original media bytes; enhanced cache generation/reuse; stale/hash-invalid or corrupt sidecar/cache regeneration; candidate tree byte/path snapshot unchanged.
4. Path traversal/encoded traversal/unknown stems/static nested paths rejected.
5. Valid draft and final PATCH, server-generated UTC first timestamp, timestamp preservation, client timestamp rejection, explicit second-review timestamp, invalid action/finalize state.
6. Revision conflict is 409 with zero mutation; validation 400; candidate change 409; mocked persistence/audit failure stable 500.
7. occupied 8765 returns 8766; a small fully occupied test range raises stable error; nonliteral loopback hosts and invalid bounds reject; sockets close.
8. No route permits system/model origin or preannotation to approve a record.

Run:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service -v`

Regression:

`D:\Anaconda3\envs\hosebot_cv\python.exe -m unittest tests.test_review_service tests.test_preannotation tests.test_review_store tests.test_review_models -v`

## Handoff

Create `.superpowers/sdd/task-4-report.md` containing RED/GREEN commands and counts, route/PATCH/cache contract, actual port-fallback test result, exact files changed, and residual concerns. Do not launch a persistent server or claim the Task 5 browser UI exists.
