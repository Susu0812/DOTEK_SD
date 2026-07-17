"""Fail-closed review validation and transactional reviewed-bundle export."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image, ImageDraw, UnidentifiedImageError

from .annotation import rasterize_centerline
from .artifacts import load_bgr, save_label, save_overlay, write_json_atomic
from .dataset_merge import snapshot_dataset
from .review_models import ReviewRecord, ReviewStatus, record_from_dict, record_to_dict, validate_record
from .review_store import ReviewStateError, ReviewStore


SAFE_STEM = re.compile(r"[A-Za-z0-9_-]+\Z")
FORMAL_SAFE_STEM = re.compile(r"[A-Za-z0-9_.-]+\Z")
EXPORT_STATUSES = frozenset({ReviewStatus.POSITIVE, ReviewStatus.HARD_NEGATIVE})
FINAL_STATUSES = frozenset(
    {ReviewStatus.POSITIVE, ReviewStatus.HARD_NEGATIVE, ReviewStatus.EXCLUDED}
)


class ReviewExportError(RuntimeError):
    """Stable fail-closed error raised when an export cannot be published."""

    def __init__(self, code: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.report = report


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, OSError):
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _safe_child(directory: Path, name: str) -> Path:
    if _is_link_like(directory):
        raise ValueError("linked_directory")
    path = directory / name
    if _is_link_like(path):
        raise ValueError("linked_child")
    if path.resolve().parent != directory.resolve():
        raise ValueError("escaped_child")
    return path


def _add_error(
    errors: list[dict[str, Any]],
    code: str,
    *,
    stem: str | None = None,
    detail: str | None = None,
) -> None:
    error: dict[str, Any] = {"code": code}
    if stem is not None:
        error["stem"] = stem
    if detail is not None:
        error["detail"] = detail
    if error not in errors:
        errors.append(error)


def _sorted_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        errors,
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("stem", "")),
            str(item.get("detail", "")),
        ),
    )


def _load_manifest(
    candidate_root: Path, errors: list[dict[str, Any]]
) -> list[dict[str, str]]:
    if _is_link_like(candidate_root):
        _add_error(errors, "candidate_root_unsafe")
        return []
    try:
        path = _safe_child(candidate_root, "manifest.csv")
        raw = path.read_text(encoding="utf-8-sig")
        if "\x00" in raw:
            raise ValueError("nul")
        reader = csv.DictReader(io.StringIO(raw, newline=""))
        fields = set(reader.fieldnames or ())
        if not {"stem", "image_sha256"}.issubset(fields):
            _add_error(errors, "manifest_missing_fields")
            return []
        rows = list(reader)
    except FileNotFoundError:
        _add_error(errors, "manifest_missing")
        return []
    except (OSError, UnicodeDecodeError, csv.Error, ValueError):
        _add_error(errors, "manifest_unreadable")
        return []

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or None in row:
            _add_error(errors, "manifest_malformed")
            continue
        stem = (row.get("stem") or "").strip()
        digest = (row.get("image_sha256") or "").strip()
        if SAFE_STEM.fullmatch(stem) is None or Path(stem).name != stem:
            _add_error(errors, "manifest_unsafe_stem", stem=stem)
            continue
        if stem in seen:
            _add_error(errors, "manifest_duplicate_stem", stem=stem)
            continue
        seen.add(stem)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _add_error(errors, "manifest_invalid_hash", stem=stem)
            continue
        records.append({"stem": stem, "image_sha256": digest})
    if not rows:
        _add_error(errors, "manifest_empty")
    return sorted(records, key=lambda item: item["stem"])


def _load_state(
    candidate_root: Path,
    work_root: Path,
    errors: list[dict[str, Any]],
) -> tuple[list[ReviewRecord], dict[str, Any] | None]:
    if _is_link_like(work_root):
        _add_error(errors, "work_root_unsafe")
        return [], None
    store = ReviewStore(work_root, candidate_root)
    try:
        summary = store.summary()
        if summary["recovered_from_last_good"]:
            _add_error(errors, "review_state_recovered")
        if summary["stale_temporary_present"]:
            _add_error(errors, "stale_temporary_state")
        if summary["audit_incomplete"]:
            _add_error(errors, "audit_incomplete")
        state, _ = store._load_state()
        records = [record_from_dict(item) for item in state["records"]]
        return records, summary
    except FileNotFoundError:
        _add_error(errors, "review_state_missing")
    except (OSError, ReviewStateError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        _add_error(errors, "review_state_invalid")
    return [], None


def _validated_candidate_bytes(
    candidate_root: Path,
    identity: dict[str, str],
    errors: list[dict[str, Any]],
) -> bytes | None:
    stem = identity["stem"]
    try:
        frames = _safe_child(candidate_root, "frames")
        path = _safe_child(frames, f"{stem}.jpg")
        content = path.read_bytes()
    except FileNotFoundError:
        _add_error(errors, "candidate_missing", stem=stem)
        return None
    except (OSError, ValueError):
        _add_error(errors, "candidate_path_unsafe", stem=stem)
        return None
    if _sha256_bytes(content) != identity["image_sha256"]:
        _add_error(errors, "candidate_hash_mismatch", stem=stem)
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "JPEG":
                _add_error(errors, "candidate_not_jpeg", stem=stem)
            if image.mode != "RGB":
                _add_error(errors, "candidate_not_rgb", stem=stem)
            if image.size != (640, 480):
                _add_error(errors, "candidate_dimensions_invalid", stem=stem)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (480, 640, 3):
                _add_error(errors, "candidate_decode_invalid", stem=stem)
    except (OSError, ValueError, UnidentifiedImageError):
        _add_error(errors, "candidate_jpeg_invalid", stem=stem)
    return content


def _scan_formal_directory(
    directory: Path,
    suffix: str,
    errors: list[dict[str, Any]],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    try:
        if _is_link_like(directory) or not directory.is_dir():
            raise ValueError("unsafe")
        for path in directory.iterdir():
            if path.suffix.lower() != suffix:
                continue
            if _is_link_like(path) or not path.is_file() or path.resolve().parent != directory.resolve():
                _add_error(errors, "formal_path_unsafe", stem=path.stem)
                continue
            if (
                FORMAL_SAFE_STEM.fullmatch(path.stem) is None
                or not path.stem
                or set(path.stem) == {"."}
            ):
                _add_error(errors, "formal_stem_unsafe", stem=path.stem)
                continue
            if path.stem in result:
                _add_error(errors, "formal_duplicate_stem", stem=path.stem)
                continue
            result[path.stem] = path
    except (OSError, ValueError):
        _add_error(errors, "formal_dataset_invalid", detail=str(directory.name))
    return result


def _formal_inventory(
    dataset_root: Path, errors: list[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    if _is_link_like(dataset_root):
        _add_error(errors, "formal_dataset_unsafe")
        return set(), set()
    all_stems: set[str] = set()
    test_hashes: set[str] = set()
    for split in ("train", "test"):
        images = _scan_formal_directory(dataset_root / split / "pic", ".jpg", errors)
        labels = _scan_formal_directory(dataset_root / split / "label", ".png", errors)
        all_stems.update(images)
        all_stems.update(labels)
        if split == "test":
            for stem, path in images.items():
                try:
                    test_hashes.add(_sha256_file(path))
                except OSError:
                    _add_error(errors, "formal_dataset_invalid", stem=stem)
    return all_stems, test_hashes


def _validate_record_and_mask(
    record: ReviewRecord, errors: list[dict[str, Any]]
) -> np.ndarray | None:
    for code in validate_record(record):
        _add_error(errors, "record_invalid", stem=record.stem, detail=code)
    if record.status is ReviewStatus.UNREVIEWED:
        _add_error(errors, "status_unreviewed", stem=record.stem)
    if record.status is ReviewStatus.NEEDS_SECOND_REVIEW:
        _add_error(errors, "status_needs_second_review", stem=record.stem)
    if record.status in FINAL_STATUSES and record.first_reviewed_at is None:
        _add_error(errors, "first_review_missing", stem=record.stem)
    if record.second_review_required and record.second_reviewed_at is None:
        _add_error(errors, "second_review_missing", stem=record.stem)

    if record.status is ReviewStatus.POSITIVE:
        try:
            label = rasterize_centerline(list(record.anchors))
            if (
                label.dtype != np.uint8
                or label.shape != (480, 640)
                or not set(int(value) for value in np.unique(label)).issubset({0, 1})
                or int(label.sum()) == 0
                or bool(np.any(label.sum(axis=1) > 1))
            ):
                raise ValueError("invalid mask")
            return label
        except Exception:
            _add_error(errors, "positive_mask_invalid", stem=record.stem)
    elif record.status is ReviewStatus.HARD_NEGATIVE:
        label = np.zeros((480, 640), dtype=np.uint8)
        if int(label.sum()) != 0:
            _add_error(errors, "hard_negative_mask_nonzero", stem=record.stem)
        return label
    return None


def _scan_bundle_files(
    root: Path,
    subdirectory: str,
    suffix: str,
    errors: list[dict[str, Any]],
) -> dict[str, Path]:
    directory = root / subdirectory
    result: dict[str, Path] = {}
    try:
        if _is_link_like(root) or _is_link_like(directory) or not directory.is_dir():
            raise ValueError("unsafe")
        for path in directory.iterdir():
            if path.suffix.lower() != suffix:
                continue
            if _is_link_like(path) or not path.is_file() or path.resolve().parent != directory.resolve():
                _add_error(errors, "bundle_path_unsafe", stem=path.stem)
                continue
            if SAFE_STEM.fullmatch(path.stem) is None:
                _add_error(errors, "bundle_stem_unsafe", stem=path.stem)
                continue
            if path.stem in result:
                _add_error(errors, "bundle_duplicate_stem", stem=path.stem)
                continue
            result[path.stem] = path
    except (OSError, ValueError):
        _add_error(errors, "bundle_directory_invalid", detail=subdirectory)
    return result


def _load_bundle_label(
    path: Path, stem: str, errors: list[dict[str, Any]]
) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "L" or image.size != (640, 480):
                raise ValueError("format")
            label = np.asarray(image)
        if label.dtype != np.uint8 or not set(int(value) for value in np.unique(label)).issubset({0, 1}):
            raise ValueError("binary")
        return label
    except (OSError, ValueError, UnidentifiedImageError):
        _add_error(errors, "bundle_label_invalid", stem=stem)
        return None


def _load_bundle_jpeg(
    path: Path,
    stem: str,
    expected_size: tuple[int, int],
    errors: list[dict[str, Any]],
) -> bytes | None:
    try:
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "JPEG" or image.mode != "RGB" or image.size != expected_size:
                raise ValueError("format")
        return content
    except (OSError, ValueError, UnidentifiedImageError):
        _add_error(errors, "bundle_jpeg_invalid", stem=stem)
        return None


def _render_overlay_jpeg(frame_bgr: np.ndarray, label: np.ndarray) -> bytes:
    visible_line = cv2.dilate(
        (label > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    overlay = frame_bgr.copy()
    overlay[visible_line] = (0, 0, 255)
    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    output = io.BytesIO()
    Image.fromarray(rgb).save(
        output,
        format="JPEG",
        quality=95,
        subsampling=0,
    )
    return output.getvalue()


def _render_contact_sheet_jpeg(overlay_root: Path, stems: list[str]) -> bytes:
    cell_width, cell_height = 320, 240
    columns, rows = 5, 4
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for offset, stem in enumerate(stems):
        row, column = divmod(offset, columns)
        with Image.open(overlay_root / f"{stem}.jpg") as image:
            image.load()
            thumbnail = image.convert("RGB").resize((cell_width, cell_height))
        x, y = column * cell_width, row * cell_height
        canvas.paste(thumbnail, (x, y))
        draw.rectangle((x, y, x + cell_width, y + 20), fill=(0, 0, 0))
        draw.text((x + 4, y + 4), stem, fill=(255, 255, 0))
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=92)
    return output.getvalue()


def _load_json_object(path: Path) -> dict[str, Any] | None:
    document = json.loads(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else None


def _report_hash(
    path: Path | None,
    errors: list[dict[str, Any]],
    detail: str,
) -> str | None:
    if path is None:
        return None
    try:
        return _sha256_file(path)
    except OSError:
        _add_error(errors, "bundle_report_unreadable", detail=detail)
        return None


def _validation_document_matches(
    document: dict[str, Any], expected: dict[str, Any]
) -> bool:
    canonical_keys = set(expected) | {"bundle_root"}
    allowed_keys = canonical_keys | {"preflight"}
    if not canonical_keys.issubset(document) or not set(document).issubset(allowed_keys):
        return False
    if any(document.get(key) != value for key, value in expected.items()):
        return False
    bundle_root = document.get("bundle_root")
    if bundle_root is not None and (
        not isinstance(bundle_root, str) or not Path(bundle_root).is_absolute()
    ):
        return False
    nested = document.get("preflight")
    if nested is not None:
        if not isinstance(nested, dict) or set(nested) != canonical_keys:
            return False
        if nested.get("bundle_root") is not None:
            return False
        if any(nested.get(key) != value for key, value in expected.items()):
            return False
    return True


def _validate_bundle(
    bundle_root: Path,
    records: list[ReviewRecord],
    identities: dict[str, dict[str, str]],
    masks: dict[str, np.ndarray],
    errors: list[dict[str, Any]],
    expected_count: int,
    expected_assignment: dict[str, Any] | None,
    formal_snapshot: dict[str, Any] | None,
    expected_validation: dict[str, Any],
) -> None:
    images = _scan_bundle_files(bundle_root, "pic", ".jpg", errors)
    labels = _scan_bundle_files(bundle_root, "label", ".png", errors)
    image_stems = set(images)
    label_stems = set(labels)
    if image_stems != label_stems:
        _add_error(errors, "bundle_pair_mismatch")
    expected = {record.stem for record in records if record.status in EXPORT_STATUSES}
    excluded = {record.stem for record in records if record.status is ReviewStatus.EXCLUDED}
    if (image_stems | label_stems) & excluded:
        for stem in sorted((image_stems | label_stems) & excluded):
            _add_error(errors, "excluded_pair_leakage", stem=stem)
    if image_stems != expected or label_stems != expected:
        _add_error(errors, "bundle_expected_stems_mismatch")

    by_stem = {record.stem: record for record in records}
    for stem in sorted(image_stems & set(identities)):
        try:
            if _sha256_file(images[stem]) != identities[stem]["image_sha256"]:
                _add_error(errors, "bundle_image_hash_mismatch", stem=stem)
        except OSError:
            _add_error(errors, "bundle_image_unreadable", stem=stem)
    loaded_labels: dict[str, np.ndarray] = {}
    for stem in sorted(label_stems & set(by_stem)):
        label = _load_bundle_label(labels[stem], stem, errors)
        if label is None:
            continue
        loaded_labels[stem] = label
        record = by_stem[stem]
        if record.status is ReviewStatus.HARD_NEGATIVE and int(label.sum()) != 0:
            _add_error(errors, "hard_negative_label_nonzero", stem=stem)
        if record.status is ReviewStatus.POSITIVE:
            if int(label.sum()) == 0 or bool(np.any(label.sum(axis=1) > 1)):
                _add_error(errors, "positive_label_invalid", stem=stem)
            expected_mask = masks.get(stem)
            if expected_mask is not None and not np.array_equal(label, expected_mask):
                _add_error(errors, "positive_label_mismatch", stem=stem)

    # A completed staging tree also contains overlays, deterministic sheets, and reports.
    overlays = _scan_bundle_files(bundle_root, "overlay", ".jpg", errors)
    if set(overlays) != expected:
        _add_error(errors, "bundle_overlay_stems_mismatch")
    for stem in sorted(set(overlays) & image_stems & set(loaded_labels)):
        overlay_content = _load_bundle_jpeg(overlays[stem], stem, (640, 480), errors)
        if overlay_content is None:
            continue
        try:
            expected_overlay = _render_overlay_jpeg(
                load_bgr(images[stem]), loaded_labels[stem]
            )
            if overlay_content != expected_overlay:
                _add_error(errors, "bundle_overlay_provenance_mismatch", stem=stem)
        except (OSError, ValueError, UnidentifiedImageError):
            _add_error(errors, "bundle_overlay_provenance_mismatch", stem=stem)
    sheets = _scan_bundle_files(bundle_root, "contact_sheets", ".jpg", errors)
    expected_sheet_names = {
        f"overlay_contact_{index:03d}"
        for index in range(1, math.ceil(len(expected) / 20) + 1)
    }
    if set(sheets) != expected_sheet_names:
        _add_error(errors, "bundle_contact_sheets_invalid")
    ordered_expected = sorted(expected)
    for page, start in enumerate(range(0, len(ordered_expected), 20), start=1):
        sheet_stem = f"overlay_contact_{page:03d}"
        path = sheets.get(sheet_stem)
        if path is None:
            continue
        content = _load_bundle_jpeg(path, sheet_stem, (1600, 960), errors)
        if content is None:
            continue
        try:
            expected_content = _render_contact_sheet_jpeg(
                bundle_root / "overlay", ordered_expected[start : start + 20]
            )
            if content != expected_content:
                _add_error(errors, "bundle_contact_sheet_provenance_mismatch", stem=sheet_stem)
        except (OSError, ValueError, UnidentifiedImageError):
            _add_error(errors, "bundle_contact_sheet_provenance_mismatch", stem=sheet_stem)

    documents: dict[str, dict[str, Any]] = {}
    for name in ("annotation.json", "review_report.json", "validation_report.json"):
        try:
            path = _safe_child(bundle_root, name)
            document = _load_json_object(path)
            if document is None:
                raise ValueError("object required")
            documents[name] = document
        except (OSError, ValueError, json.JSONDecodeError):
            _add_error(errors, "bundle_report_invalid", detail=name)

    annotation = documents.get("annotation.json")
    expected_records = [record_to_dict(record) for record in records]
    if annotation is not None and annotation != {
        "schema_version": 1,
        "records": expected_records,
    }:
        _add_error(errors, "bundle_report_mismatch", detail="annotation.json")

    review = documents.get("review_report.json")
    if review is not None:
        mismatch = (
            review.get("schema_version") != 1
            or review.get("exported_count") != len(expected)
            or review.get("excluded_count") != len(excluded)
            or review.get("exported_stems") != sorted(expected)
            or review.get("excluded_stems") != sorted(excluded)
            or expected_assignment is None
            or review.get("second_review_assignment") != expected_assignment
            or formal_snapshot is None
            or review.get("formal_dataset_before") != formal_snapshot
            or review.get("formal_dataset_after") != formal_snapshot
        )
        file_hashes = review.get("file_hashes")
        if not isinstance(file_hashes, dict) or set(file_hashes) != expected:
            mismatch = True
        else:
            for stem in sorted(expected):
                item = file_hashes.get(stem)
                paths = {
                    "image_sha256": images.get(stem),
                    "label_sha256": labels.get(stem),
                    "overlay_sha256": overlays.get(stem),
                }
                if not isinstance(item, dict) or set(item) != set(paths):
                    mismatch = True
                    continue
                for key, path in paths.items():
                    actual_hash = _report_hash(path, errors, f"{stem}:{key}")
                    if actual_hash is None or item.get(key) != actual_hash:
                        mismatch = True
        contact_hashes = review.get("contact_sheet_hashes")
        expected_contact_files = {path.name: path for path in sheets.values()}
        if not isinstance(contact_hashes, dict) or set(contact_hashes) != set(expected_contact_files):
            mismatch = True
        else:
            for name, path in expected_contact_files.items():
                actual_hash = _report_hash(path, errors, name)
                if actual_hash is None or contact_hashes.get(name) != actual_hash:
                    mismatch = True
        if mismatch:
            _add_error(errors, "bundle_report_mismatch", detail="review_report.json")

    validation = documents.get("validation_report.json")
    if validation is not None and not _validation_document_matches(
        validation, expected_validation
    ):
        _add_error(errors, "bundle_report_mismatch", detail="validation_report.json")


def validate_review(
    candidate_root: Path,
    work_root: Path,
    dataset_root: Path,
    expected_count: int = 388,
    bundle_root: Path | None = None,
    seed: int = 20260716,
) -> dict[str, Any]:
    """Return a deterministic, read-only report containing every detected error."""

    candidate_root = Path(candidate_root)
    work_root = Path(work_root)
    dataset_root = Path(dataset_root)
    errors: list[dict[str, Any]] = []
    manifest = _load_manifest(candidate_root, errors)
    if len(manifest) != expected_count:
        _add_error(
            errors,
            "expected_count_mismatch",
            detail=f"expected={expected_count},actual={len(manifest)}",
        )
    records, summary = _load_state(candidate_root, work_root, errors)
    manifest_identity = [(item["stem"], item["image_sha256"]) for item in manifest]
    state_identity = sorted((record.stem, record.image_sha256) for record in records)
    if state_identity != manifest_identity:
        _add_error(errors, "state_identity_mismatch")

    identities = {item["stem"]: item for item in manifest}
    candidate_hashes: dict[str, str] = {}
    for item in manifest:
        content = _validated_candidate_bytes(candidate_root, item, errors)
        if content is not None:
            candidate_hashes[item["stem"]] = _sha256_bytes(content)

    masks: dict[str, np.ndarray] = {}
    for record in records:
        mask = _validate_record_and_mask(record, errors)
        if mask is not None:
            masks[record.stem] = mask

    formal_stems, formal_test_hashes = _formal_inventory(dataset_root, errors)
    for stem in sorted(set(identities) & formal_stems):
        _add_error(errors, "formal_stem_collision", stem=stem)
    for stem, digest in sorted(candidate_hashes.items()):
        if digest in formal_test_hashes:
            _add_error(errors, "formal_test_hash_collision", stem=stem)

    try:
        formal_snapshot: dict[str, Any] | None = snapshot_dataset(dataset_root).to_dict()
    except (OSError, ValueError, FileNotFoundError):
        formal_snapshot = None
        _add_error(errors, "formal_dataset_invalid")
    try:
        expected_assignment: dict[str, Any] | None = _second_review_plan(
            records, _historically_required(work_root), seed
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        expected_assignment = None
        _add_error(errors, "audit_incomplete")

    status_counts = {status.value: 0 for status in ReviewStatus}
    for record in records:
        status_counts[record.status.value] += 1
    expected_validation = {
        "schema_version": 1,
        "ok": True,
        "errors": [],
        "error_codes": [],
        "seed": seed,
        "expected_count": expected_count,
        "manifest_count": len(manifest),
        "record_count": len(records),
        "status_counts": status_counts,
        "candidate_stems": sorted(identities),
        "audit_summary": summary,
    }

    if bundle_root is not None:
        _validate_bundle(
            Path(bundle_root),
            records,
            identities,
            masks,
            errors,
            expected_count,
            expected_assignment,
            formal_snapshot,
            expected_validation,
        )

    errors = _sorted_errors(errors)
    return {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "error_codes": sorted({item["code"] for item in errors}),
        "seed": seed,
        "expected_count": expected_count,
        "manifest_count": len(manifest),
        "record_count": len(records),
        "status_counts": status_counts,
        "candidate_stems": sorted(identities),
        "bundle_root": str(Path(bundle_root).resolve()) if bundle_root is not None else None,
        "audit_summary": summary,
    }


def _warning_reasons(warnings: tuple[str, ...]) -> set[str]:
    reasons: set[str] = set()
    for warning in warnings:
        normalized = re.sub(r"[^a-z0-9]+", "_", warning.casefold())
        if "temporal" in normalized:
            reasons.add("warning_temporal_disagreement")
        if (
            ("prediction" in normalized or "model" in normalized)
            and ("disagreement" in normalized or "conflict" in normalized)
        ):
            reasons.add("warning_prediction_model_conflict")
        if "low_confidence" in normalized or (
            "low" in normalized and "confidence" in normalized
        ):
            reasons.add("warning_low_confidence")
        if "occlusion" in normalized or "occluded" in normalized:
            reasons.add("warning_occlusion")
    return reasons


def _historically_required(work_root: Path) -> set[str]:
    path = work_root / "annotation_history.jsonl"
    if not path.exists():
        return set()
    required: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            stem = event["stem"]
            for value in (event["before"], event["after"]):
                if value.get("status") == ReviewStatus.NEEDS_SECOND_REVIEW.value or value.get(
                    "second_review_required"
                ) is True:
                    required.add(stem)
    return required


def _second_review_plan(
    records: list[ReviewRecord], historical: set[str], seed: int
) -> dict[str, Any]:
    intrinsic: dict[str, set[str]] = {record.stem: set() for record in records}
    reasons: dict[str, set[str]] = {record.stem: set() for record in records}
    for record in records:
        if record.status is ReviewStatus.HARD_NEGATIVE:
            intrinsic[record.stem].add("hard_negative")
        if record.status is ReviewStatus.POSITIVE and record.suggestion_modified:
            intrinsic[record.stem].add("suggestion_modified")
        intrinsic[record.stem].update(_warning_reasons(record.warnings))
        if record.interference_tags:
            intrinsic[record.stem].add("interference_tags")
        reasons[record.stem].update(intrinsic[record.stem])
        if record.stem in historical:
            reasons[record.stem].add("audit_history")

    mandatory = sorted(stem for stem, values in reasons.items() if values)
    # Audit history is mandatory, but it does not change whether a current
    # positive is intrinsically plain. Keeping that sample universe stable is
    # what makes repeated audited assignments idempotent without discarding any
    # qualifying history event.
    plain_positive = sorted(
        record.stem
        for record in records
        if record.status is ReviewStatus.POSITIVE and not intrinsic[record.stem]
    )
    sample_count = math.ceil(len(plain_positive) * 0.10)
    sampled = sorted(
        sorted(
            plain_positive,
            key=lambda stem: (hashlib.sha256(f"{seed}:{stem}".encode()).hexdigest(), stem),
        )[:sample_count]
    )
    for stem in sampled:
        reasons[stem].add("deterministic_audit_sample")
    selected = set(mandatory) | set(sampled)
    already = sorted(record.stem for record in records if record.second_review_required)
    by_stem = {record.stem: record for record in records}
    newly = sorted(stem for stem in selected if not by_stem[stem].second_review_required)
    return {
        "schema_version": 1,
        "seed": seed,
        "mandatory_stems": mandatory,
        "sampled_stems": sampled,
        "already_required_stems": already,
        "newly_required_stems": newly,
        "reasons": {
            stem: sorted(values) for stem, values in sorted(reasons.items()) if values
        },
    }


def _assignment_context(
    candidate_root: Path, work_root: Path, seed: int
) -> tuple[list[ReviewRecord], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    manifest = _load_manifest(candidate_root, errors)
    records, summary = _load_state(candidate_root, work_root, errors)
    if summary is None or summary.get("recovered_from_last_good") or summary.get(
        "stale_temporary_present"
    ) or summary.get("audit_incomplete"):
        _add_error(errors, "assignment_state_not_clean")
    if sorted((record.stem, record.image_sha256) for record in records) != [
        (item["stem"], item["image_sha256"]) for item in manifest
    ]:
        _add_error(errors, "state_identity_mismatch")
    for item in manifest:
        _validated_candidate_bytes(candidate_root, item, errors)
    try:
        historical = _historically_required(work_root)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        historical = set()
        _add_error(errors, "audit_incomplete")
    if errors:
        codes = ",".join(sorted({item["code"] for item in errors}))
        raise ReviewStateError(f"second-review assignment refused: {codes}")
    return records, _second_review_plan(records, historical, seed)


def assign_second_review_requirements(
    candidate_root: Path,
    work_root: Path,
    seed: int = 20260716,
) -> dict[str, Any]:
    """Monotonically assign mandatory and deterministic audit second reviews."""

    candidate_root = Path(candidate_root)
    work_root = Path(work_root)
    records, report = _assignment_context(candidate_root, work_root, seed)
    by_stem = {record.stem: record for record in records}
    store = ReviewStore(work_root, candidate_root)
    for stem in report["newly_required_stems"]:
        record = by_stem[stem]
        updated = store.update(
            stem,
            {"second_review_required": True},
            record.revision,
            actor="second_review_assignment",
            origin="human",
        )
        by_stem[stem] = updated
    return report


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_safe_output_location(
    output_root: Path, staging: Path, protected_roots: tuple[Path, ...]
) -> None:
    if _is_link_like(output_root) or _is_link_like(staging):
        raise ReviewExportError("unsafe_output_location")
    current = output_root.parent
    while True:
        if current.exists() and _is_link_like(current):
            raise ReviewExportError("unsafe_output_location")
        if current.parent == current:
            break
        current = current.parent
    for protected in protected_roots:
        if _path_within(output_root, protected) or _path_within(staging, protected):
            raise ReviewExportError("unsafe_output_location")


def _write_contact_sheets(staging: Path, stems: list[str]) -> dict[str, str]:
    contact_root = staging / "contact_sheets"
    contact_root.mkdir()
    hashes: dict[str, str] = {}
    per_sheet = 20
    for page, start in enumerate(range(0, len(stems), per_sheet), start=1):
        name = f"overlay_contact_{page:03d}.jpg"
        content = _render_contact_sheet_jpeg(
            staging / "overlay", stems[start : start + per_sheet]
        )
        (contact_root / name).write_bytes(content)
        hashes[name] = _sha256_bytes(content)
    return hashes


def export_reviewed_bundle(
    candidate_root: Path,
    work_root: Path,
    dataset_root: Path,
    output_root: Path,
    expected_count: int = 388,
    seed: int = 20260716,
) -> Path:
    """Export reviewed pairs transactionally without modifying the formal dataset."""

    candidate_root = Path(candidate_root)
    work_root = Path(work_root)
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    staging = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or _is_link_like(output_root):
        raise ReviewExportError("output_exists")
    if staging.exists() or _is_link_like(staging):
        raise ReviewExportError("stale_staging")
    protected_roots = (candidate_root, work_root, dataset_root)
    _assert_safe_output_location(output_root, staging, protected_roots)

    preflight = validate_review(
        candidate_root,
        work_root,
        dataset_root,
        expected_count=expected_count,
        seed=seed,
    )
    if not preflight["ok"]:
        raise ReviewExportError("preflight_failed", preflight)
    try:
        _, assignment = _assignment_context(candidate_root, work_root, seed)
    except ReviewStateError as exc:
        raise ReviewExportError("preflight_failed", preflight) from exc
    if assignment["newly_required_stems"]:
        raise ReviewExportError("second_review_assignment_required", assignment)
    try:
        formal_before = snapshot_dataset(dataset_root)
    except Exception as exc:
        raise ReviewExportError("formal_snapshot_failed") from exc

    store = ReviewStore(work_root, candidate_root)
    state, recovered = store._load_state()
    if recovered:
        raise ReviewExportError("preflight_failed")
    records = [record_from_dict(item) for item in state["records"]]
    exported = sorted(record.stem for record in records if record.status in EXPORT_STATUSES)
    excluded = sorted(record.stem for record in records if record.status is ReviewStatus.EXCLUDED)
    by_stem = {record.stem: record for record in records}

    try:
        for name in ("pic", "label", "overlay"):
            (staging / name).mkdir(parents=True, exist_ok=name != "pic")
        file_hashes: dict[str, dict[str, str]] = {}
        for stem in exported:
            record = by_stem[stem]
            source = _safe_child(_safe_child(candidate_root, "frames"), f"{stem}.jpg")
            destination = staging / "pic" / f"{stem}.jpg"
            shutil.copyfile(source, destination)
            copied_hash = _sha256_file(destination)
            if copied_hash != record.image_sha256 or copied_hash != _sha256_file(source):
                raise ReviewExportError("copied_image_hash_mismatch")
            if record.status is ReviewStatus.POSITIVE:
                label = rasterize_centerline(list(record.anchors))
            else:
                label = np.zeros((480, 640), dtype=np.uint8)
            label_path = staging / "label" / f"{stem}.png"
            overlay_path = staging / "overlay" / f"{stem}.jpg"
            save_label(label_path, label)
            save_overlay(overlay_path, load_bgr(destination), label)
            file_hashes[stem] = {
                "image_sha256": copied_hash,
                "label_sha256": _sha256_file(label_path),
                "overlay_sha256": _sha256_file(overlay_path),
            }
        contact_sheet_hashes = _write_contact_sheets(staging, exported)

        annotation = {
            "schema_version": 1,
            "records": [record_to_dict(record) for record in records],
        }
        formal_before_dict = formal_before.to_dict()
        review_report = {
            "schema_version": 1,
            "exported_count": len(exported),
            "excluded_count": len(excluded),
            "exported_stems": exported,
            "excluded_stems": excluded,
            "file_hashes": file_hashes,
            "contact_sheet_hashes": contact_sheet_hashes,
            "second_review_assignment": assignment,
            "formal_dataset_before": formal_before_dict,
            "formal_dataset_after": formal_before_dict,
        }
        write_json_atomic(staging / "annotation.json", annotation)
        write_json_atomic(staging / "review_report.json", review_report)
        write_json_atomic(staging / "validation_report.json", preflight)

        staging_validation = validate_review(
            candidate_root,
            work_root,
            dataset_root,
            expected_count=expected_count,
            bundle_root=staging,
            seed=seed,
        )
        if not staging_validation["ok"]:
            raise ReviewExportError("staging_validation_failed", staging_validation)
        formal_after = snapshot_dataset(dataset_root)
        if formal_after != formal_before:
            raise ReviewExportError("formal_dataset_changed")
        review_report["formal_dataset_after"] = formal_after.to_dict()
        write_json_atomic(staging / "review_report.json", review_report)
        final_validation = dict(staging_validation)
        final_validation["preflight"] = preflight
        write_json_atomic(staging / "validation_report.json", final_validation)

        publish_validation = validate_review(
            candidate_root,
            work_root,
            dataset_root,
            expected_count=expected_count,
            bundle_root=staging,
            seed=seed,
        )
        if not publish_validation["ok"]:
            raise ReviewExportError("staging_validation_failed", publish_validation)
        if snapshot_dataset(dataset_root) != formal_before:
            raise ReviewExportError("formal_dataset_changed")
        if output_root.exists():
            raise ReviewExportError("output_exists")
        _assert_safe_output_location(output_root, staging, protected_roots)
        os.replace(staging, output_root)
        return output_root
    except ReviewExportError:
        raise
    except Exception as exc:
        raise ReviewExportError("export_failed") from exc
