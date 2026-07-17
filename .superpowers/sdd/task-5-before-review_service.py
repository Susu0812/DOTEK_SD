"""Loopback-only WSGI service for the local human review workbench.

The module deliberately does not start a server.  ``create_app`` returns a WSGI
application for a caller (or Werkzeug's test client), while
``find_available_port`` performs a short-lived loopback bind probe only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import socket
import stat
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, MethodNotAllowed, NotFound, UnsupportedMediaType
from werkzeug.routing import Map, Rule
from werkzeug.wrappers import Request, Response

from low_light_dataset.image_ops import enhance_low_light, save_jpeg
from low_light_dataset.review_models import record_to_dict
from low_light_dataset.review_store import (
    AuditError,
    CandidateChangedError,
    ReviewStateError,
    ReviewStore,
    RevisionConflict,
)


WSGIApplication = Callable[
    [dict[str, Any], Callable[[str, list[tuple[str, str]]], Any]], Iterable[bytes]
]

FINAL_STATUSES = frozenset({"positive", "hard_negative", "excluded"})
QUEUE_STATUSES = frozenset({"unreviewed", "needs_second_review"})
PATCH_FIELDS = frozenset(
    {
        "status",
        "anchors",
        "interference_tags",
        "exclusion_reason",
        "warnings",
        "notes",
    }
)
CLIENT_TIMESTAMP_FIELDS = frozenset({"first_reviewed_at", "second_reviewed_at"})
CONTROL_FIELDS = frozenset({"revision", "action", "actor"})
PREVIEW_SCHEMA_VERSION = 1
SAFE_STEM_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")
PREVIEW_PARAM_FIELDS = frozenset(
    {
        "gamma",
        "retinex_weight",
        "clahe_clip_limit",
        "denoise_sigma_color",
        "sharpen_amount",
    }
)


def _json_response(payload: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        status=status,
        content_type="application/json; charset=utf-8",
    )


def _error(code: str, status: int) -> Response:
    return _json_response({"error": code}, status)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")
    if "\x00" in raw:
        raise ValueError("unsafe_manifest_stem")
    try:
        reader = csv.DictReader(io.StringIO(raw, newline=""))
        if reader.fieldnames is None:
            raise ValueError("manifest missing header")
        rows: list[dict[str, Any]] = [dict(row) for row in reader]
    except csv.Error as exc:
        raise ValueError("invalid manifest") from exc

    def chronological_key(row: dict[str, Any]) -> tuple[int, float, str]:
        raw = row.get("target_timestamp_seconds")
        try:
            timestamp = float(raw)
        except (TypeError, ValueError):
            return (1, 0.0, str(row.get("stem", "")))
        return (0, timestamp, str(row.get("stem", "")))

    rows.sort(key=chronological_key)
    for row in rows:
        raw = row.get("target_timestamp_seconds")
        if raw is not None:
            try:
                row["target_timestamp_seconds"] = float(raw)
            except (TypeError, ValueError):
                pass
    return rows


def _validate_manifest_stems(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        stem = row.get("stem")
        if not isinstance(stem, str):
            raise ValueError("unsafe_manifest_stem")
        candidate = Path(stem)
        if (
            stem in {"", ".", ".."}
            or "/" in stem
            or "\\" in stem
            or candidate.name != stem
            or candidate.is_absolute()
            or bool(candidate.drive)
            or bool(candidate.root)
            or any(ord(character) < 32 or ord(character) == 127 for character in stem)
            or SAFE_STEM_PATTERN.fullmatch(stem) is None
        ):
            raise ValueError("unsafe_manifest_stem")


def _load_preannotations(path: Path | None, stems: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ValueError("invalid preannotation document")
    suggestions: dict[str, dict[str, Any]] = {}
    for item in document["records"]:
        if not isinstance(item, dict):
            raise ValueError("invalid preannotation record")
        stem = item.get("stem")
        if isinstance(stem, str) and stem in stems and stem not in suggestions:
            # The whole suggestion remains visibly non-authoritative because it is
            # nested.  It can never overwrite the human record at the top level.
            suggestions[stem] = dict(item)
    return suggestions


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _is_link_like(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without following them."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _resolved_direct_child(directory: Path, name: str) -> Path:
    """Return a lexical child only when its resolved target stays direct."""

    if _is_link_like(directory):
        raise ValueError("unsafe linked directory")
    path = directory / name
    if _is_link_like(path):
        raise ValueError("unsafe linked child")
    resolved_directory = directory.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_directory:
        raise ValueError("unsafe resolved child")
    return path


class ReviewApplication:
    """Directly testable WSGI application returned by :func:`create_app`.

    ``review_store`` is intentionally public for diagnostics and focused tests;
    request authorization remains fixed to a human local reviewer.
    """

    def __init__(
        self,
        candidate_root: Path,
        work_root: Path,
        preannotation_path: Path | None,
        static_root: Path | None,
    ) -> None:
        self.candidate_root = Path(candidate_root)
        self.work_root = Path(work_root)
        if _path_is_within(self.work_root, self.candidate_root):
            raise ValueError("work_root must not be inside candidate_root")
        self.static_root = Path(static_root) if static_root is not None else None
        self.manifest_path = self.candidate_root / "manifest.csv"
        self.manifest_records = _load_manifest_rows(self.manifest_path)
        _validate_manifest_stems(self.manifest_records)
        self._manifest_by_stem = {
            str(row["stem"]): row for row in self.manifest_records
        }
        self._ordered_stems = tuple(str(row["stem"]) for row in self.manifest_records)
        self._stem_set = set(self._ordered_stems)
        if len(self._stem_set) != len(self._ordered_stems):
            raise ValueError("duplicate manifest stem")
        self.frames_root = self.candidate_root / "frames"
        for stem in self._ordered_stems:
            self._resolved_candidate_path(stem)
        self._safe_cache_root()
        self.review_store = ReviewStore(self.work_root, self.candidate_root)
        state = self.review_store.initialize(self.manifest_path)
        state_stems = {str(item["stem"]) for item in state["records"]}
        if state_stems != self._stem_set:
            raise ReviewStateError("existing state manifest does not match")
        self.preannotations = _load_preannotations(preannotation_path, self._stem_set)
        self._preview_locks = {stem: threading.Lock() for stem in self._ordered_stems}
        self.url_map = Map(
            [
                Rule("/health", endpoint="health", methods=["GET"]),
                Rule("/api/summary", endpoint="summary", methods=["GET"]),
                Rule("/api/records", endpoint="records", methods=["GET"]),
                Rule(
                    "/api/records/<stem>",
                    endpoint="record",
                    methods=["GET", "PATCH"],
                ),
                Rule("/api/review-queue", endpoint="queue", methods=["GET"]),
                Rule(
                    "/media/original/<stem>.jpg",
                    endpoint="original",
                    methods=["GET"],
                ),
                Rule(
                    "/media/enhanced/<stem>.jpg",
                    endpoint="enhanced",
                    methods=["GET"],
                ),
                Rule("/static/<name>", endpoint="static", methods=["GET"]),
            ]
        )

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]):
        request = Request(environ)
        response = self._dispatch_safely(request)
        return response(environ, start_response)

    def _dispatch_safely(self, request: Request) -> Response:
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match(method=request.method)
        except NotFound:
            return _error("not_found", 404)
        except MethodNotAllowed:
            return _error("method_not_allowed", 405)
        if request.method == "HEAD":
            return _error("method_not_allowed", 405)

        try:
            if endpoint == "health":
                summary = self.review_store.summary()
                return _json_response(
                    {
                        "ok": True,
                        "candidate_count": len(self._ordered_stems),
                        "record_count": summary["total"],
                    }
                )
            if endpoint == "summary":
                return _json_response(self.review_store.summary())
            if endpoint == "records":
                return _json_response({"records": self._joined_records()})
            if endpoint == "record":
                stem = values["stem"]
                if stem not in self._stem_set:
                    return _error("not_found", 404)
                if request.method == "GET":
                    return _json_response(self._detail(stem))
                return self._patch_record(request, stem)
            if endpoint == "queue":
                records = [
                    item
                    for item in self._joined_records()
                    if item["status"] in QUEUE_STATUSES
                    or (
                        item["second_review_required"]
                        and item["second_reviewed_at"] is None
                    )
                ]
                return _json_response(
                    {"stems": [item["stem"] for item in records], "records": records}
                )
            if endpoint == "original":
                return self._original(values["stem"])
            if endpoint == "enhanced":
                return self._enhanced(values["stem"])
            if endpoint == "static":
                return self._static(values["name"])
            return _error("not_found", 404)
        except CandidateChangedError:
            return _error("candidate_changed", 409)
        except RevisionConflict:
            return _error("revision_conflict", 409)
        except KeyError:
            return _error("not_found", 404)
        except (AuditError, ReviewStateError, OSError):
            return _error("save_failed", 500)
        except Exception:
            # A local service must not leak tracebacks or filesystem paths to its
            # HTTP response.  Initialization errors still propagate from create_app.
            return _error("service_failed", 500)

    def _joined(self, stem: str) -> dict[str, Any]:
        record = record_to_dict(self.review_store.get(stem))
        joined = dict(self._manifest_by_stem[stem])
        joined.update(record)
        suggestion = self.preannotations.get(stem)
        if suggestion is not None:
            joined["preannotation"] = dict(suggestion)
        return joined

    def _joined_records(self) -> list[dict[str, Any]]:
        return [self._joined(stem) for stem in self._ordered_stems]

    def _detail(self, stem: str) -> dict[str, Any]:
        index = self._ordered_stems.index(stem)
        detail = self._joined(stem)
        detail["previous_stem"] = self._ordered_stems[index - 1] if index else None
        detail["next_stem"] = (
            self._ordered_stems[index + 1]
            if index + 1 < len(self._ordered_stems)
            else None
        )
        return detail

    def _patch_record(self, request: Request, stem: str) -> Response:
        try:
            payload = request.get_json()
        except (BadRequest, UnsupportedMediaType):
            return _error("invalid_json", 400)
        if not isinstance(payload, dict):
            return _error("invalid_json", 400)
        if CLIENT_TIMESTAMP_FIELDS.intersection(payload):
            return _error("client_timestamp_forbidden", 400)

        revision = payload.get("revision")
        if not _is_integer(revision) or revision < 0:
            return _error("invalid_revision", 400)
        action = payload.get("action", "draft")
        if not isinstance(action, str) or action not in {
            "draft",
            "finalize",
            "second_review",
        }:
            return _error("invalid_action", 400)

        patch = {key: value for key, value in payload.items() if key not in CONTROL_FIELDS}
        if not set(patch).issubset(PATCH_FIELDS):
            return _error("invalid_patch", 400)

        before = self.review_store.get(stem)
        effective_status = patch.get("status", before.status.value)
        is_final_status = (
            isinstance(effective_status, str) and effective_status in FINAL_STATUSES
        )
        if action in {"finalize", "second_review"} and not is_final_status:
            return _error("final_status_required", 400)

        now = _utc_now()
        if effective_status == "needs_second_review":
            patch["second_review_required"] = True
        if is_final_status and before.first_reviewed_at is None:
            patch["first_reviewed_at"] = now
        if action == "second_review":
            patch["second_review_required"] = True
            patch["second_reviewed_at"] = now

        try:
            after = self.review_store.update(
                stem,
                patch,
                revision,
                actor="local_web_reviewer",
                origin="human",
            )
        except RevisionConflict:
            return _error("revision_conflict", 409)
        except CandidateChangedError:
            return _error("candidate_changed", 409)
        except (ValueError, TypeError, PermissionError):
            return _error("validation_failed", 400)
        except (AuditError, ReviewStateError, OSError):
            return _error("save_failed", 500)
        return _json_response(record_to_dict(after))

    def _canonical_path(self, stem: str) -> Path:
        if stem not in self._stem_set:
            raise KeyError(stem)
        path = self._resolved_candidate_path(stem)
        self.review_store.get(stem)
        # Recheck after hashing so a link swap cannot silently become the path read.
        if self._resolved_candidate_path(stem) != path:
            raise CandidateChangedError(stem)
        return path

    def _resolved_candidate_path(self, stem: str) -> Path:
        try:
            return _resolved_direct_child(self.frames_root, f"{stem}.jpg")
        except (OSError, ValueError) as exc:
            raise CandidateChangedError(stem) from exc

    def _safe_cache_root(self) -> Path:
        if _is_link_like(self.work_root):
            raise ReviewStateError("unsafe preview cache path")
        cache_root = self.work_root / "enhanced_preview_cache"
        if _is_link_like(cache_root):
            raise ReviewStateError("unsafe preview cache path")
        try:
            if cache_root.resolve().parent != self.work_root.resolve():
                raise ReviewStateError("unsafe preview cache path")
        except OSError as exc:
            raise ReviewStateError("unsafe preview cache path") from exc
        return cache_root

    def _preview_paths(self, stem: str) -> tuple[Path, Path]:
        cache_root = self._safe_cache_root()
        try:
            cache_path = _resolved_direct_child(cache_root, f"{stem}.jpg")
            sidecar_path = _resolved_direct_child(cache_root, f"{stem}.json")
        except (OSError, ValueError) as exc:
            raise ReviewStateError("unsafe preview cache path") from exc
        return cache_path, sidecar_path

    def _original(self, stem: str) -> Response:
        path = self._canonical_path(stem)
        return Response(path.read_bytes(), content_type="image/jpeg")

    def _enhanced(self, stem: str) -> Response:
        source_path = self._canonical_path(stem)
        source_hash = self._manifest_by_stem[stem]["image_sha256"]
        cache_path, sidecar_path = self._preview_paths(stem)
        with self._preview_locks[stem]:
            cache_path, sidecar_path = self._preview_paths(stem)
            if not self._valid_cached_preview(cache_path, sidecar_path, source_hash):
                self._regenerate_preview(
                    source_path,
                    source_hash,
                    cache_path,
                    sidecar_path,
                    stem,
                )
            cache_path, _ = self._preview_paths(stem)
            return Response(cache_path.read_bytes(), content_type="image/jpeg")

    @staticmethod
    def _decode_jpeg(content: bytes) -> tuple[np.ndarray, str]:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "JPEG" or image.size != (640, 480):
                raise ValueError("JPEG must be 640x480")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy(), "jpeg"

    def _valid_cached_preview(
        self, cache_path: Path, sidecar_path: Path, source_hash: str
    ) -> bool:
        try:
            content = cache_path.read_bytes()
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                return False
            if (
                metadata.get("schema_version") != PREVIEW_SCHEMA_VERSION
                or metadata.get("source_sha256") != source_hash
            ):
                return False
            params = metadata.get("enhancement_parameters")
            if (
                not isinstance(params, dict)
                or set(params) != PREVIEW_PARAM_FIELDS
                or not all(_is_finite_number(value) for value in params.values())
            ):
                return False
            output = metadata.get("output")
            if not isinstance(output, dict) or output != {
                "format": "jpeg",
                "width": 640,
                "height": 480,
                "byte_size": len(content),
                "sha256": _sha256_bytes(content),
            }:
                return False
            self._decode_jpeg(content)
            return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, UnidentifiedImageError):
            return False

    def _regenerate_preview(
        self,
        source_path: Path,
        source_hash: str,
        cache_path: Path,
        sidecar_path: Path,
        stem: str,
    ) -> None:
        source, _ = self._decode_jpeg(source_path.read_bytes())
        enhanced = enhance_low_light(source)
        cache_root = self._safe_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_root = self._safe_cache_root()
        verified_cache_path, verified_sidecar_path = self._preview_paths(stem)
        if verified_cache_path != cache_path or verified_sidecar_path != sidecar_path:
            raise ReviewStateError("unsafe preview cache path")
        jpeg_temporary: Path | None = None
        json_temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=cache_root,
                prefix=f".{stem}.",
                suffix=".jpg.tmp",
            )
            os.close(descriptor)
            jpeg_temporary = Path(temporary_name)
            save_jpeg(jpeg_temporary, enhanced.image)
            output = jpeg_temporary.read_bytes()
            self._decode_jpeg(output)
            params = enhanced.params.to_dict()
            metadata = {
                "schema_version": PREVIEW_SCHEMA_VERSION,
                "source_sha256": source_hash,
                "enhancement_parameters": params,
                "output": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "byte_size": len(output),
                    "sha256": _sha256_bytes(output),
                },
            }

            descriptor, temporary_name = tempfile.mkstemp(
                dir=cache_root,
                prefix=f".{stem}.",
                suffix=".json.tmp",
            )
            json_temporary = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            verified_cache_path, verified_sidecar_path = self._preview_paths(stem)
            if verified_cache_path != cache_path or verified_sidecar_path != sidecar_path:
                raise ReviewStateError("unsafe preview cache path")
            os.replace(jpeg_temporary, cache_path)
            jpeg_temporary = None
            os.replace(json_temporary, sidecar_path)
            json_temporary = None
        finally:
            if jpeg_temporary is not None:
                jpeg_temporary.unlink(missing_ok=True)
            if json_temporary is not None:
                json_temporary.unlink(missing_ok=True)

    def _static(self, name: str) -> Response:
        if self.static_root is None:
            return _error("not_found", 404)
        if not name or Path(name).name != name or "/" in name or "\\" in name:
            return _error("not_found", 404)
        path = self.static_root / name
        if not path.is_file() or path.resolve().parent != self.static_root.resolve():
            return _error("not_found", 404)
        mimetype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(path.read_bytes(), content_type=mimetype)


def create_app(
    candidate_root: Path,
    work_root: Path,
    preannotation_path: Path | None = None,
    static_root: Path | None = None,
) -> WSGIApplication:
    """Initialize compatible review state and return the loopback WSGI app.

    Existing incompatible state is rejected by :class:`ReviewStore`; the service
    never replaces it silently and never starts a persistent HTTP server.
    """

    return ReviewApplication(
        Path(candidate_root),
        Path(work_root),
        Path(preannotation_path) if preannotation_path is not None else None,
        Path(static_root) if static_root is not None else None,
    )


def find_available_port(
    host: str = "127.0.0.1",
    start: int = 8765,
    end: int = 8799,
) -> int:
    """Return the first bindable port in an inclusive literal-loopback range.

    Every probe socket is closed before return.  This function reserves no port;
    the Task 8 CLI must bind the returned port itself.
    """

    if host != "127.0.0.1":
        raise ValueError("host_must_be_literal_127.0.0.1")
    if (
        not _is_integer(start)
        or not _is_integer(end)
        or not 1 <= start <= 65535
        or not 1 <= end <= 65535
        or start > end
    ):
        raise ValueError("invalid_port_bounds")
    for port in range(start, end + 1):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
        except OSError:
            continue
        finally:
            probe.close()
        return port
    raise RuntimeError(f"no_loopback_port_available_{start}_{end}")
