"""Safe snapshots and transactional merge for the formal dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import validate_prepared_bundle, write_json_atomic


@dataclass(frozen=True)
class SplitSnapshot:
    images: dict[str, str]
    labels: dict[str, str]

    def to_dict(self) -> dict[str, dict[str, str]]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSnapshot:
    train: SplitSnapshot
    test: SplitSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {"train": self.train.to_dict(), "test": self.test.to_dict()}


@dataclass(frozen=True)
class MergePlan:
    bundle_root: Path
    dataset_root: Path
    before: DatasetSnapshot
    new_images: tuple[Path, ...]
    new_labels: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == suffix.lower():
            if path.stem in result:
                raise ValueError(f"duplicate stem in {directory}: {path.stem}")
            result[path.stem] = path
    return result


def snapshot_split(split_root: Path) -> SplitSnapshot:
    images = _files_by_stem(split_root / "pic", ".jpg")
    labels = _files_by_stem(split_root / "label", ".png")
    return SplitSnapshot(
        images={stem: sha256_file(path) for stem, path in sorted(images.items())},
        labels={stem: sha256_file(path) for stem, path in sorted(labels.items())},
    )


def snapshot_dataset(dataset_root: Path) -> DatasetSnapshot:
    return DatasetSnapshot(
        train=snapshot_split(dataset_root / "train"),
        test=snapshot_split(dataset_root / "test"),
    )


def save_snapshot(dataset_root: Path, output_path: Path) -> Path:
    snapshot = snapshot_dataset(dataset_root)
    document = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "snapshot": snapshot.to_dict(),
    }
    write_json_atomic(output_path, document)
    return output_path


def preflight_merge(
    bundle_root: Path,
    dataset_root: Path,
    *,
    expected_new_count: int = 60,
    expected_train_before: int = 4858,
    expected_test_count: int = 867,
) -> MergePlan:
    validation = validate_prepared_bundle(
        bundle_root,
        expected_count=expected_new_count,
        require_review=True,
    )
    if not validation.ok:
        raise ValueError("bundle validation failed: " + "; ".join(validation.errors))

    before = snapshot_dataset(dataset_root)
    if set(before.train.images) != set(before.train.labels):
        raise ValueError("existing training image/label stems do not match")
    if set(before.test.images) != set(before.test.labels):
        raise ValueError("existing test image/label stems do not match")
    if len(before.train.images) != expected_train_before:
        raise ValueError(
            f"training baseline is not {expected_train_before}/{expected_train_before}"
        )
    if len(before.test.images) != expected_test_count:
        raise ValueError(f"test baseline is not {expected_test_count}/{expected_test_count}")

    new_images_by_stem = _files_by_stem(bundle_root / "enhanced", ".jpg")
    new_labels_by_stem = _files_by_stem(bundle_root / "label", ".png")
    new_stems = set(new_images_by_stem)
    if new_stems != set(new_labels_by_stem):
        raise ValueError("new image/label stems do not match")
    existing_stems = set(before.train.images) | set(before.test.images)
    collisions = sorted(new_stems & existing_stems)
    if collisions:
        raise ValueError(f"name collision: {collisions}")

    test_hashes = set(before.test.images.values())
    leaked = sorted(
        stem
        for stem, path in new_images_by_stem.items()
        if sha256_file(path) in test_hashes
    )
    if leaked:
        raise ValueError(f"test-set leakage by image hash: {leaked}")

    return MergePlan(
        bundle_root=bundle_root,
        dataset_root=dataset_root,
        before=before,
        new_images=tuple(new_images_by_stem[stem] for stem in sorted(new_stems)),
        new_labels=tuple(new_labels_by_stem[stem] for stem in sorted(new_stems)),
    )


def _old_hashes_unchanged(before: DatasetSnapshot, after: DatasetSnapshot) -> bool:
    for stem, digest in before.train.images.items():
        if after.train.images.get(stem) != digest:
            return False
    for stem, digest in before.train.labels.items():
        if after.train.labels.get(stem) != digest:
            return False
    return before.test == after.test


def merge_bundle(
    bundle_root: Path,
    dataset_root: Path,
    *,
    expected_new_count: int = 60,
    expected_train_before: int = 4858,
    expected_test_count: int = 867,
) -> Path:
    plan = preflight_merge(
        bundle_root,
        dataset_root,
        expected_new_count=expected_new_count,
        expected_train_before=expected_train_before,
        expected_test_count=expected_test_count,
    )
    operations: list[tuple[Path, Path, Path]] = []
    for source in plan.new_images:
        final = dataset_root / "train" / "pic" / source.name
        temporary = final.with_name(f".{final.name}.lowlight-staging")
        operations.append((source, temporary, final))
    for source in plan.new_labels:
        final = dataset_root / "train" / "label" / source.name
        temporary = final.with_name(f".{final.name}.lowlight-staging")
        operations.append((source, temporary, final))

    published: list[Path] = []
    try:
        for source, temporary, _ in operations:
            if temporary.exists():
                raise FileExistsError(f"stale merge temporary exists: {temporary}")
            shutil.copyfile(source, temporary)
            if sha256_file(source) != sha256_file(temporary):
                raise IOError(f"hash mismatch after copy: {source.name}")
        for _, temporary, final in operations:
            temporary.replace(final)
            published.append(final)

        after = snapshot_dataset(dataset_root)
        expected_after = expected_train_before + expected_new_count
        if len(after.train.images) != expected_after or len(after.train.labels) != expected_after:
            raise RuntimeError(
                f"post-merge training count is not {expected_after}/{expected_after}"
            )
        if len(after.test.images) != expected_test_count or len(after.test.labels) != expected_test_count:
            raise RuntimeError("test split changed during merge")
        if set(after.train.images) != set(after.train.labels):
            raise RuntimeError("post-merge training stems do not match")
        if not _old_hashes_unchanged(plan.before, after):
            raise RuntimeError("an existing dataset file changed during merge")

        receipt_path = bundle_root / "merge_receipt.json"
        receipt = {
            "schema_version": 1,
            "dataset_root": str(dataset_root.resolve()),
            "before_counts": {
                "train_images": len(plan.before.train.images),
                "train_labels": len(plan.before.train.labels),
                "test_images": len(plan.before.test.images),
                "test_labels": len(plan.before.test.labels),
            },
            "after_counts": {
                "train_images": len(after.train.images),
                "train_labels": len(after.train.labels),
                "test_images": len(after.test.images),
                "test_labels": len(after.test.labels),
            },
            "new_files": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in published
            ],
        }
        write_json_atomic(receipt_path, receipt)
        return receipt_path
    except Exception:
        for _, temporary, _ in operations:
            if temporary.exists():
                temporary.unlink()
        for path in published:
            if path.exists() and path.stem.startswith("lowlight_camera_full_rgb_"):
                path.unlink()
        raise


def verify_merge(
    bundle_root: Path,
    dataset_root: Path,
    baseline_path: Path,
    *,
    expected_new_count: int = 60,
) -> dict[str, Any]:
    baseline_document = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_data = baseline_document["snapshot"]
    baseline = DatasetSnapshot(
        train=SplitSnapshot(**baseline_data["train"]),
        test=SplitSnapshot(**baseline_data["test"]),
    )
    current = snapshot_dataset(dataset_root)
    errors: list[str] = []
    if not _old_hashes_unchanged(baseline, current):
        errors.append("baseline files changed")
    if len(current.train.images) != len(baseline.train.images) + expected_new_count:
        errors.append("unexpected final training image count")
    if set(current.train.images) != set(current.train.labels):
        errors.append("final training stems do not match")
    receipt_path = bundle_root / "merge_receipt.json"
    if not receipt_path.is_file():
        errors.append("missing merge receipt")
    return {
        "ok": not errors,
        "errors": errors,
        "train_images": len(current.train.images),
        "train_labels": len(current.train.labels),
        "test_images": len(current.test.images),
        "test_labels": len(current.test.labels),
    }
