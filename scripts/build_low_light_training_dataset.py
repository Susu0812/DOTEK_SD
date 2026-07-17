"""Build and safely merge the reviewed low-light hose training dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_light_dataset.artifacts import (
    finalize_labels,
    prepare_dataset,
    validate_prepared_bundle,
)
from low_light_dataset.dataset_merge import (
    merge_bundle,
    save_snapshot,
    snapshot_dataset,
    verify_merge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare reviewed low-light hose samples without training a model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="extract, enhance, and pre-label 60 frames")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")

    finalize = commands.add_parser("finalize", help="apply explicit review decisions")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--review", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate reviewed image/label pairs")
    validate.add_argument("--output", type=Path, required=True)

    snapshot = commands.add_parser("snapshot", help="record the existing formal dataset")
    snapshot.add_argument("--dataset", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    merge = commands.add_parser("merge", help="transactionally add reviewed pairs to train")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--dataset", type=Path, required=True)

    verify = commands.add_parser("verify-merge", help="verify final counts and original hashes")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--baseline", type=Path, required=True)
    return parser


def _print(document: object) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare":
        path = prepare_dataset(
            arguments.video,
            arguments.output,
            arguments.checkpoint,
            arguments.device,
        )
        _print({"ok": True, "output": str(path.resolve())})
        return 0
    if arguments.command == "finalize":
        report = finalize_labels(arguments.output, arguments.review)
        _print(report.to_dict())
        return 0 if report.ok else 1
    if arguments.command == "validate":
        report = validate_prepared_bundle(arguments.output)
        _print(report.to_dict())
        return 0 if report.ok else 1
    if arguments.command == "snapshot":
        path = save_snapshot(arguments.dataset, arguments.output)
        snapshot = snapshot_dataset(arguments.dataset)
        _print(
            {
                "ok": True,
                "output": str(path.resolve()),
                "train_images": len(snapshot.train.images),
                "train_labels": len(snapshot.train.labels),
                "test_images": len(snapshot.test.images),
                "test_labels": len(snapshot.test.labels),
            }
        )
        return 0
    if arguments.command == "merge":
        receipt = merge_bundle(arguments.output, arguments.dataset)
        _print({"ok": True, "receipt": str(receipt.resolve())})
        return 0
    if arguments.command == "verify-merge":
        report = verify_merge(arguments.output, arguments.dataset, arguments.baseline)
        _print(report)
        return 0 if report["ok"] else 1
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    sys.exit(main())
