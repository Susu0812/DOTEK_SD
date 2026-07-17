"""Safely quarantine a previously merged low-light image/label bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_merge import snapshot_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _counts(dataset_root: Path) -> dict[str, int]:
    snapshot = snapshot_dataset(dataset_root)
    return {
        "train_images": len(snapshot.train.images),
        "train_labels": len(snapshot.train.labels),
        "test_images": len(snapshot.test.images),
        "test_labels": len(snapshot.test.labels),
    }


def _validate_final_counts(
    dataset_root: Path, expected_train_after: int, expected_test_count: int
) -> dict[str, int]:
    snapshot = snapshot_dataset(dataset_root)
    if set(snapshot.train.images) != set(snapshot.train.labels):
        raise RuntimeError("formal train image/label stems are not paired")
    if set(snapshot.test.images) != set(snapshot.test.labels):
        raise RuntimeError("formal test image/label stems are not paired")
    counts = _counts(dataset_root)
    expected = {
        "train_images": expected_train_after,
        "train_labels": expected_train_after,
        "test_images": expected_test_count,
        "test_labels": expected_test_count,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected formal dataset counts: {counts}, expected {expected}")
    return counts


def _receipt_entries(
    bundle_root: Path, dataset_root: Path, expected_count: int
) -> tuple[Path, list[tuple[Path, str, str]]]:
    receipt_path = bundle_root / "merge_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded_root = Path(document.get("dataset_root", "")).resolve()
    if recorded_root != dataset_root.resolve():
        raise ValueError(
            f"merge receipt dataset root mismatch: {recorded_root} != {dataset_root.resolve()}"
        )
    image_root = (dataset_root / "train" / "pic").resolve()
    label_root = (dataset_root / "train" / "label").resolve()
    entries: list[tuple[Path, str, str]] = []
    for item in document.get("new_files", []):
        path = Path(item["path"]).resolve()
        digest = str(item["sha256"]).lower()
        if path.parent == image_root and path.suffix.lower() == ".jpg":
            kind = "train_pic"
        elif path.parent == label_root and path.suffix.lower() == ".png":
            kind = "train_label"
        else:
            raise ValueError(f"receipt path is outside formal train pair folders: {path}")
        entries.append((path, digest, kind))
    images = [entry for entry in entries if entry[2] == "train_pic"]
    labels = [entry for entry in entries if entry[2] == "train_label"]
    if len(images) != expected_count or len(labels) != expected_count:
        raise ValueError(
            f"expected {expected_count} receipt pairs, got {len(images)} images and {len(labels)} labels"
        )
    if {path.stem for path, _, _ in images} != {path.stem for path, _, _ in labels}:
        raise ValueError("receipt image/label stems are not paired")
    return receipt_path, entries


def _validate_existing_quarantine(
    quarantine_root: Path,
    dataset_root: Path,
    expected_count: int,
    expected_train_after: int,
    expected_test_count: int,
) -> Path:
    receipt_path = quarantine_root / "quarantine_receipt.json"
    if not receipt_path.is_file():
        raise FileExistsError(
            f"quarantine exists without a complete receipt: {quarantine_root}"
        )
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    files = document.get("files", [])
    if len(files) != expected_count * 2:
        raise RuntimeError("existing quarantine receipt has the wrong file count")
    for item in files:
        destination = quarantine_root / item["relative_destination"]
        if not destination.is_file() or _sha256(destination) != item["sha256"]:
            raise RuntimeError(f"existing quarantine file validation failed: {destination}")
    _validate_final_counts(dataset_root, expected_train_after, expected_test_count)
    return receipt_path


def quarantine_merged_bundle(
    bundle_root: Path,
    dataset_root: Path,
    quarantine_root: Path,
    expected_count: int = 60,
    expected_train_after: int = 4858,
    expected_test_count: int = 867,
) -> Path:
    """Move a hash-verified merge out of formal train and retain an audit receipt."""

    bundle_root = Path(bundle_root)
    dataset_root = Path(dataset_root)
    quarantine_root = Path(quarantine_root)
    if quarantine_root.exists():
        return _validate_existing_quarantine(
            quarantine_root, dataset_root, expected_count,
            expected_train_after, expected_test_count,
        )
    staging = quarantine_root.with_name(quarantine_root.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"incomplete quarantine staging exists: {staging}")

    receipt_path, entries = _receipt_entries(
        bundle_root, dataset_root, expected_count
    )
    before_counts = _counts(dataset_root)
    for source, expected_hash, _ in entries:
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"hash mismatch for {source}: {actual_hash} != {expected_hash}"
            )

    (staging / "train_pic").mkdir(parents=True)
    (staging / "train_label").mkdir()
    file_records: list[dict[str, str]] = []
    for source, expected_hash, kind in entries:
        destination = staging / kind / source.name
        shutil.copy2(source, destination)
        if _sha256(destination) != expected_hash:
            raise RuntimeError(f"staging copy hash mismatch: {destination}")
        file_records.append(
            {
                "source": str(source),
                "relative_destination": f"{kind}/{source.name}",
                "sha256": expected_hash,
            }
        )

    removed: list[tuple[Path, Path]] = []
    try:
        for source, _, kind in entries:
            staged = staging / kind / source.name
            source.unlink()
            removed.append((source, staged))
        after_counts = _validate_final_counts(
            dataset_root, expected_train_after, expected_test_count
        )
    except Exception:
        for source, staged in removed:
            if not source.exists() and staged.is_file():
                shutil.copy2(staged, source)
        raise

    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "bundle_root": str(bundle_root.resolve()),
        "merge_receipt": str(receipt_path.resolve()),
        "merge_receipt_sha256": _sha256(receipt_path),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "files": file_records,
    }
    _write_json(staging / "source_bundle_manifest.json", {
        "bundle_root": str(bundle_root.resolve()),
        "merge_receipt_sha256": document["merge_receipt_sha256"],
    })
    _write_json(staging / "quarantine_receipt.json", document)
    quarantine_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, quarantine_root)
    return quarantine_root / "quarantine_receipt.json"

