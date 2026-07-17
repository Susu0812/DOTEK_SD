"""Quarantine the flawed merge and extract a new unlabeled candidate set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from low_light_dataset.candidates import extract_candidate_set
from low_light_dataset.dataset_merge import snapshot_dataset
from low_light_dataset.quarantine import quarantine_merged_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely refresh low-light video candidates without labels or training."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    quarantine = commands.add_parser("quarantine")
    quarantine.add_argument("--bundle", type=Path, required=True)
    quarantine.add_argument("--dataset", type=Path, required=True)
    quarantine.add_argument("--output", type=Path, required=True)
    quarantine.add_argument("--expected-count", type=int, default=60)
    quarantine.add_argument("--expected-train-after", type=int, default=4858)
    quarantine.add_argument("--expected-test-count", type=int, default=867)

    extract = commands.add_parser("extract")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--duration", type=float, default=300.0)
    extract.add_argument("--interval", type=float, default=0.25)

    verify = commands.add_parser("verify")
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--quarantine", type=Path, required=True)
    verify.add_argument("--candidates", type=Path, required=True)
    verify.add_argument("--expected-train", type=int, default=4858)
    verify.add_argument("--expected-test", type=int, default=867)
    return parser


def _print(document: object) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


def verify_outputs(
    dataset: Path, quarantine: Path, candidates: Path,
    expected_train: int, expected_test: int,
) -> dict[str, object]:
    snapshot = snapshot_dataset(dataset)
    errors: list[str] = []
    if len(snapshot.train.images) != expected_train or len(snapshot.train.labels) != expected_train:
        errors.append("formal train count mismatch")
    if len(snapshot.test.images) != expected_test or len(snapshot.test.labels) != expected_test:
        errors.append("formal test count mismatch")
    if set(snapshot.train.images) != set(snapshot.train.labels):
        errors.append("formal train pairing mismatch")
    if not (quarantine / "quarantine_receipt.json").is_file():
        errors.append("missing quarantine receipt")
    summary_path = candidates / "summary.json"
    manifest_path = candidates / "manifest.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        errors.append("missing candidate summary or manifest")
        candidate_count = 0
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            candidate_count = sum(1 for _ in csv.DictReader(handle))
        if candidate_count != summary.get("candidate_count"):
            errors.append("candidate manifest count mismatch")
    if list(candidates.rglob("*.png")):
        errors.append("candidate directory unexpectedly contains labels")
    return {
        "ok": not errors,
        "errors": errors,
        "train_images": len(snapshot.train.images),
        "train_labels": len(snapshot.train.labels),
        "test_images": len(snapshot.test.images),
        "test_labels": len(snapshot.test.labels),
        "candidate_count": candidate_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "quarantine":
        receipt = quarantine_merged_bundle(
            args.bundle, args.dataset, args.output,
            expected_count=args.expected_count,
            expected_train_after=args.expected_train_after,
            expected_test_count=args.expected_test_count,
        )
        _print({"ok": True, "receipt": str(receipt.resolve())})
        return 0
    if args.command == "extract":
        summary = extract_candidate_set(
            args.video, args.output, args.duration, args.interval
        )
        _print({"ok": True, "summary": str(summary.resolve())})
        return 0
    if args.command == "verify":
        report = verify_outputs(
            args.dataset, args.quarantine, args.candidates,
            args.expected_train, args.expected_test,
        )
        _print(report)
        return 0 if report["ok"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())

