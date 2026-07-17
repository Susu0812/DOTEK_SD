"""Atomic persistence and audit history for human review records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from low_light_dataset.review_models import (
    ReviewRecord,
    ReviewStatus,
    record_from_dict,
    record_to_dict,
    validate_record,
)


SCHEMA_VERSION = 2
ALLOWED_ORIGINS = frozenset({"human", "system", "model"})
NONHUMAN_ORIGINS = frozenset({"system", "model"})
FINAL_STATUSES = frozenset(
    {
        ReviewStatus.POSITIVE.value,
        ReviewStatus.HARD_NEGATIVE.value,
        ReviewStatus.EXCLUDED.value,
    }
)
EDITABLE_KEYS = frozenset(
    {
        "status",
        "anchors",
        "interference_tags",
        "exclusion_reason",
        "warnings",
        "suggestion_modified",
        "first_reviewed_at",
        "second_review_required",
        "second_reviewed_at",
        "notes",
    }
)

_UPDATE_LOCKS: dict[str, threading.RLock] = {}
_UPDATE_LOCKS_GUARD = threading.Lock()


class RevisionConflict(RuntimeError):
    """Raised when an update is based on an obsolete record revision."""


class CandidateChangedError(RuntimeError):
    """Raised when a candidate JPEG no longer matches its manifest hash."""


class ReviewStateError(RuntimeError):
    """Raised when persisted review state is corrupt or incompatible."""


class AuditError(RuntimeError):
    """Raised when state is published but its audit event cannot be appended."""


class ReviewStore:
    def __init__(self, work_root: Path, candidate_root: Path) -> None:
        self.work_root = Path(work_root)
        self.candidate_root = Path(candidate_root)
        self.state_path = self.work_root / "annotation_state.json"
        self.backup_path = self.work_root / "annotation_state.json.last-good"
        self.temporary_path = self.work_root / "annotation_state.json.tmp"
        self.history_path = self.work_root / "annotation_history.jsonl"
        self.lock_path = self.work_root / "annotation_state.lock"
        lock_key = str(self.work_root.resolve()).casefold()
        with _UPDATE_LOCKS_GUARD:
            self._update_lock = _UPDATE_LOCKS.setdefault(lock_key, threading.RLock())
        self._audit_append_failed = False

    def initialize(self, manifest_path: Path) -> dict[str, Any]:
        manifest_records = self._read_manifest(Path(manifest_path))
        desired_identity = [
            (item["stem"], item["image_sha256"]) for item in manifest_records
        ]

        if self.state_path.exists():
            state, _ = self._load_state()
            current_identity = [
                (item["stem"], item["image_sha256"]) for item in state["records"]
            ]
            if current_identity != desired_identity:
                raise ReviewStateError("existing state manifest does not match")
            return state

        state = {
            "schema_version": SCHEMA_VERSION,
            "records": [self._new_record(item) for item in manifest_records],
        }
        self._write_state(state)
        return state

    def summary(self) -> dict[str, Any]:
        state, recovered = self._load_state()
        records = [record_from_dict(item) for item in state["records"]]
        counts = {status.value: 0 for status in ReviewStatus}
        for record in records:
            counts[record.status.value] += 1
        return {
            "schema_version": state["schema_version"],
            "total": len(records),
            "counts": counts,
            "recovered_from_last_good": recovered,
            "stale_temporary_present": self.temporary_path.exists(),
            "audit_incomplete": self._audit_append_failed
            or self._history_is_incomplete(records, recovered),
        }

    def get(self, stem: str) -> ReviewRecord:
        state, _ = self._load_state()
        _, record = self._find_record(state, stem)
        self._verify_candidate(record)
        return record

    def update(
        self,
        stem: str,
        patch: dict[str, Any],
        expected_revision: int,
        actor: str = "human",
        origin: str = "human",
    ) -> ReviewRecord:
        with self._update_guard():
            return self._update_locked(
                stem,
                patch,
                expected_revision,
                actor=actor,
                origin=origin,
            )

    def _update_locked(
        self,
        stem: str,
        patch: dict[str, Any],
        expected_revision: int,
        actor: str,
        origin: str,
    ) -> ReviewRecord:
        state, recovered = self._load_state()
        if recovered:
            raise ReviewStateError("update rejected from recovered review state")
        index, before = self._find_record(state, stem)
        self._verify_candidate(before)
        if before.revision != expected_revision:
            raise RevisionConflict(stem)
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("actor must be a non-empty string")
        if not isinstance(origin, str) or origin not in ALLOWED_ORIGINS:
            raise ValueError("origin must be human, system, or model")
        if not isinstance(patch, dict):
            raise ValueError("patch must be a dictionary")
        forbidden = set(patch).difference(EDITABLE_KEYS)
        if forbidden:
            raise ValueError("patch keys must be editable fields")
        merged = record_to_dict(before)
        merged.update(patch)
        merged["revision"] = before.revision + 1
        try:
            after = record_from_dict(merged)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid review record: {exc}") from exc
        if not self._origin_can_publish(origin, after):
            raise PermissionError(
                "non-human origin cannot publish a finalized or reviewed record"
            )
        errors = validate_record(after)
        if errors:
            raise ValueError("invalid review record: " + ", ".join(errors))

        new_state = {
            "schema_version": SCHEMA_VERSION,
            "records": list(state["records"]),
        }
        new_state["records"][index] = record_to_dict(after)
        self._write_state(new_state)

        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "origin": origin,
            "stem": stem,
            "prior_revision": before.revision,
            "new_revision": after.revision,
            "before": record_to_dict(before),
            "after": record_to_dict(after),
        }
        try:
            self._append_history(event)
        except Exception as exc:
            self._audit_append_failed = True
            raise AuditError(str(exc)) from exc
        return after

    @contextmanager
    def _update_guard(self):
        with self._update_lock:
            self.work_root.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                self._acquire_file_lock(handle)
                try:
                    yield
                finally:
                    self._release_file_lock(handle)

    @staticmethod
    def _acquire_file_lock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _release_file_lock(handle) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_manifest(self, manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {
                "stem",
                "image_sha256",
            }.issubset(reader.fieldnames):
                raise ValueError("manifest missing required columns")
            rows = list(reader)
        if not rows:
            raise ValueError("empty manifest")

        records: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            stem = row["stem"]
            expected_hash = row["image_sha256"]
            if stem in seen:
                raise ValueError(f"duplicate stem: {stem}")
            seen.add(stem)
            candidate_path = self._candidate_path(stem)
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            if self._sha256(candidate_path) != expected_hash:
                raise CandidateChangedError(stem)
            records.append({"stem": stem, "image_sha256": expected_hash})
        return sorted(records, key=lambda item: item["stem"])

    @staticmethod
    def _new_record(identity: dict[str, str]) -> dict[str, Any]:
        return {
            "stem": identity["stem"],
            "image_sha256": identity["image_sha256"],
            "revision": 0,
            "status": ReviewStatus.UNREVIEWED.value,
            "anchors": [],
            "interference_tags": [],
            "exclusion_reason": None,
            "warnings": [],
            "suggestion_modified": False,
            "first_reviewed_at": None,
            "second_review_required": False,
            "second_reviewed_at": None,
            "notes": "",
        }

    def _load_state(self) -> tuple[dict[str, Any], bool]:
        if not self.state_path.exists():
            raise FileNotFoundError(self.state_path)
        try:
            return self._parse_state(self.state_path.read_bytes()), False
        except (OSError, ReviewStateError):
            pass
        if self.backup_path.exists():
            try:
                return self._parse_state(self.backup_path.read_bytes()), True
            except (OSError, ReviewStateError):
                pass
        raise ReviewStateError("no valid review state")

    @staticmethod
    def _parse_state(raw: bytes) -> dict[str, Any]:
        try:
            state = json.loads(raw.decode("utf-8"))
            if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
                raise ReviewStateError("incompatible review state")
            items = state.get("records")
            if not isinstance(items, list) or not items:
                raise ReviewStateError("incompatible review state")
            stems: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ReviewStateError("incompatible review state")
                record = record_from_dict(item)
                if validate_record(record):
                    raise ReviewStateError("invalid review record")
                stems.append(record.stem)
            if len(stems) != len(set(stems)) or stems != sorted(stems):
                raise ReviewStateError("incompatible review state")
            return state
        except ReviewStateError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ReviewStateError("invalid review state") from exc

    def _write_state(self, state: dict[str, Any]) -> None:
        self._parse_state(self._serialized_state(state))
        self.work_root.mkdir(parents=True, exist_ok=True)
        serialized = self._serialized_state(state)
        with self.temporary_path.open("wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        if self.state_path.exists():
            current = self.state_path.read_bytes()
            try:
                self._parse_state(current)
            except ReviewStateError:
                pass
            else:
                with self.backup_path.open("wb") as backup:
                    backup.write(current)
                    backup.flush()
                    os.fsync(backup.fileno())
        self.temporary_path.replace(self.state_path)

    @staticmethod
    def _serialized_state(state: dict[str, Any]) -> bytes:
        return (
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def _append_history(self, event: dict[str, Any]) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _history_is_incomplete(
        self, records: list[ReviewRecord], recovered: bool
    ) -> bool:
        current = {record.stem: record for record in records}
        if not self.history_path.exists():
            return any(record.revision > 0 for record in records)

        required_fields = {
            "time",
            "actor",
            "origin",
            "stem",
            "prior_revision",
            "new_revision",
            "before",
            "after",
        }
        last_revision = {stem: 0 for stem in current}
        last_after: dict[str, dict[str, Any]] = {}
        matches_current = {
            stem: record.revision == 0 for stem, record in current.items()
        }
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not required_fields.issubset(event):
                        return True
                    if not self._valid_audit_metadata(event):
                        return True

                    stem = event["stem"]
                    if not isinstance(stem, str) or stem not in current:
                        return True
                    prior = event["prior_revision"]
                    new = event["new_revision"]
                    if (
                        not self._is_revision(prior)
                        or not self._is_revision(new)
                        or prior != last_revision[stem]
                        or new != prior + 1
                    ):
                        return True

                    before = self._audit_record(event["before"])
                    after = self._audit_record(event["after"])
                    if before is None or after is None:
                        return True
                    if not self._origin_can_publish(event["origin"], after):
                        return True
                    expected_hash = current[stem].image_sha256
                    if (
                        before.stem != stem
                        or after.stem != stem
                        or before.image_sha256 != expected_hash
                        or after.image_sha256 != expected_hash
                        or before.revision != prior
                        or after.revision != new
                    ):
                        return True

                    before_dict = record_to_dict(before)
                    after_dict = record_to_dict(after)
                    if stem in last_after:
                        if before_dict != last_after[stem]:
                            return True
                    elif before_dict != self._new_record(
                        {"stem": stem, "image_sha256": expected_hash}
                    ):
                        return True

                    last_revision[stem] = new
                    last_after[stem] = after_dict
                    if new == current[stem].revision:
                        matches_current[stem] = after_dict == record_to_dict(current[stem])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return True

        for stem, record in current.items():
            if last_revision[stem] < record.revision or not matches_current[stem]:
                return True
            if not recovered and last_revision[stem] != record.revision:
                return True
        return False

    @staticmethod
    def _valid_audit_metadata(event: dict[str, Any]) -> bool:
        actor = event["actor"]
        origin = event["origin"]
        timestamp = event["time"]
        if not isinstance(actor, str) or not actor.strip():
            return False
        if (
            not isinstance(origin, str)
            or origin not in ALLOWED_ORIGINS
            or not isinstance(timestamp, str)
        ):
            return False
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)

    @staticmethod
    def _is_revision(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _origin_can_publish(origin: str, after: ReviewRecord) -> bool:
        return origin not in NONHUMAN_ORIGINS or (
            after.status.value not in FINAL_STATUSES
            and after.first_reviewed_at is None
        )

    @staticmethod
    def _audit_record(value: Any) -> ReviewRecord | None:
        if not isinstance(value, dict):
            return None
        try:
            record = record_from_dict(value)
            if validate_record(record):
                return None
            return record
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _find_record(
        state: dict[str, Any], stem: str
    ) -> tuple[int, ReviewRecord]:
        for index, item in enumerate(state["records"]):
            if item["stem"] == stem:
                return index, record_from_dict(item)
        raise KeyError(stem)

    def _verify_candidate(self, record: ReviewRecord) -> None:
        path = self._candidate_path(record.stem)
        if not path.is_file() or self._sha256(path) != record.image_sha256:
            raise CandidateChangedError(record.stem)

    def _candidate_path(self, stem: str) -> Path:
        return self.candidate_root / "frames" / f"{stem}.jpg"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
