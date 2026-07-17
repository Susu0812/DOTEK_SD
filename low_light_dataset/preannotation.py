"""Read-only model-assisted suggestions for low-light review candidates."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .annotation import (
    AnchorPrediction,
    HoseAnnotator,
    extract_color_anchors,
    fuse_predictions,
    regularize_anchors,
)
from .image_ops import enhance_low_light


SCHEMA_VERSION = 1
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
_REQUIRED_MANIFEST_FIELDS = {
    "stem",
    "target_timestamp_seconds",
    "image_sha256",
}


@dataclass(frozen=True)
class _Candidate:
    stem: str
    target_timestamp_seconds: float
    image_sha256: str
    image_path: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_output_location(
    candidate_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    resolved_root = candidate_root.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_root or resolved_root in resolved_output.parents:
        raise ValueError("output_path must be outside candidate_root")
    resolved_checkpoint = checkpoint_path.resolve()
    if resolved_checkpoint == resolved_output:
        raise ValueError("output_path must not alias checkpoint_path")


def _load_manifest(candidate_root: Path) -> list[_Candidate]:
    manifest_path = candidate_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_MANIFEST_FIELDS - fields)
        if missing:
            raise ValueError(f"manifest missing required fields: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("manifest contains no candidate records")

    candidates: list[_Candidate] = []
    seen_stems: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"malformed manifest row {row_index}")
        stem = (row.get("stem") or "").strip()
        timestamp_text = (row.get("target_timestamp_seconds") or "").strip()
        image_sha256 = (row.get("image_sha256") or "").strip()
        if not stem or Path(stem).name != stem or "/" in stem or "\\" in stem:
            raise ValueError(f"invalid stem in manifest row {row_index}")
        if stem in seen_stems:
            raise ValueError(f"duplicate manifest stem: {stem}")
        seen_stems.add(stem)
        try:
            timestamp = float(timestamp_text)
        except ValueError as error:
            raise ValueError(
                f"invalid target_timestamp_seconds in manifest row {row_index}"
            ) from error
        if not math.isfinite(timestamp):
            raise ValueError(
                f"invalid target_timestamp_seconds in manifest row {row_index}"
            )
        if not _is_hex_sha256(image_sha256):
            raise ValueError(f"invalid image_sha256 in manifest row {row_index}")
        candidates.append(
            _Candidate(
                stem=stem,
                target_timestamp_seconds=timestamp,
                image_sha256=image_sha256,
                image_path=candidate_root / "frames" / f"{stem}.jpg",
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (candidate.target_timestamp_seconds, candidate.stem),
    )


def _load_validated_jpeg(candidate: _Candidate) -> np.ndarray:
    if not candidate.image_path.is_file():
        raise FileNotFoundError(candidate.image_path)
    data = candidate.image_path.read_bytes()
    actual_hash = _sha256_bytes(data)
    if actual_hash != candidate.image_sha256:
        raise ValueError(f"candidate JPEG hash mismatch: {candidate.stem}")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError(f"candidate is not a JPEG: {candidate.stem}")
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"unreadable candidate JPEG: {candidate.stem}") from error

    encoded = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"unreadable candidate JPEG: {candidate.stem}")
    if frame.dtype != np.uint8 or frame.shape != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        3,
    ):
        raise ValueError(
            f"candidate JPEG must be {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}: {candidate.stem}"
        )
    return frame


def _unpack_predictions(
    value: Any,
) -> tuple[list[AnchorPrediction], list[str]]:
    warnings: Iterable[Any] = ()
    predictions = value
    if isinstance(value, tuple) and len(value) == 2:
        predictions, warnings = value
    if not isinstance(predictions, list) or not all(
        isinstance(point, AnchorPrediction) for point in predictions
    ):
        raise TypeError("prediction source must return AnchorPrediction values")
    return predictions, _normalize_warnings(warnings)


def _normalize_warnings(warnings: Iterable[Any]) -> list[str]:
    normalized: set[str] = set()
    for warning in warnings:
        if not isinstance(warning, str) or not warning:
            raise TypeError("warnings must be non-empty strings")
        normalized.add(warning)
    return sorted(normalized)


def _serialize_anchor(anchor: AnchorPrediction) -> dict[str, int | float | str]:
    y = int(anchor.y)
    x = float(anchor.x)
    confidence = float(anchor.confidence)
    if y != anchor.y or not math.isfinite(x) or not math.isfinite(confidence):
        raise ValueError("anchor coordinates and confidence must be finite")
    if not isinstance(anchor.source, str) or not anchor.source:
        raise ValueError("anchor source must be a non-empty string")
    return {
        "y": y,
        "x": x,
        "confidence": confidence,
        "source": anchor.source,
    }


def _serialize_metrics(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
    elif is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError("enhancement metrics must be serializable mappings")
    if not isinstance(result, dict):
        raise TypeError("enhancement metrics must serialize to a mapping")
    return result


def _anchor_map(record: Mapping[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    anchors = record.get("anchors", [])
    if not isinstance(anchors, list):
        return result
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        y = anchor.get("y")
        x = anchor.get("x")
        if isinstance(y, bool) or not isinstance(y, int):
            continue
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            continue
        x_float = float(x)
        if math.isfinite(x_float) and y not in result:
            result[y] = x_float
    return result


def add_temporal_warnings(
    records: list[dict[str, Any]],
    threshold_px: float = 96.0,
) -> list[dict[str, Any]]:
    """Add audit-only temporal warnings without changing any anchor geometry."""

    if isinstance(threshold_px, bool) or not isinstance(threshold_px, (int, float)):
        raise TypeError("threshold_px must be numeric")
    threshold = float(threshold_px)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold_px must be finite and non-negative")

    copied = copy.deepcopy(records)
    decorated: list[tuple[float, str, dict[str, Any]]] = []
    for index, record in enumerate(copied):
        if not isinstance(record, dict):
            raise TypeError("records must contain dictionaries")
        timestamp = record.get("target_timestamp_seconds")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("record target_timestamp_seconds must be numeric")
        timestamp_float = float(timestamp)
        if not math.isfinite(timestamp_float):
            raise ValueError("record target_timestamp_seconds must be finite")
        stem = record.get("stem")
        tie_breaker = stem if isinstance(stem, str) else f"{index:012d}"
        record_warnings = record.get("warnings", [])
        if not isinstance(record_warnings, list):
            raise TypeError("record warnings must be a list")
        warnings = _normalize_warnings(record_warnings)
        anchors = record.get("anchors", [])
        if not isinstance(anchors, list):
            raise TypeError("record anchors must be a list")
        if len(anchors) < 3:
            warnings = _normalize_warnings([*warnings, "insufficient_anchor_count"])
        record["warnings"] = warnings
        decorated.append((timestamp_float, tie_breaker, record))

    decorated.sort(key=lambda item: (item[0], item[1]))
    ordered = [item[2] for item in decorated]
    timestamps = [item[0] for item in decorated]

    for current_index, current in enumerate(ordered):
        current_time = timestamps[current_index]
        earlier_index = next(
            (
                index
                for index in range(current_index - 1, -1, -1)
                if timestamps[index] < current_time
            ),
            None,
        )
        later_index = next(
            (
                index
                for index in range(current_index + 1, len(ordered))
                if timestamps[index] > current_time
            ),
            None,
        )
        if earlier_index is None or later_index is None:
            continue
        earlier_time = timestamps[earlier_index]
        later_time = timestamps[later_index]
        time_span = later_time - earlier_time
        if time_span <= 0.0:
            continue

        earlier_by_y = _anchor_map(ordered[earlier_index])
        current_by_y = _anchor_map(current)
        later_by_y = _anchor_map(ordered[later_index])
        shared_rows = sorted(set(earlier_by_y) & set(current_by_y) & set(later_by_y))
        if len(shared_rows) < 3:
            continue
        fraction = (current_time - earlier_time) / time_span
        residuals = [
            abs(
                current_by_y[y]
                - (
                    earlier_by_y[y]
                    + fraction * (later_by_y[y] - earlier_by_y[y])
                )
            )
            for y in shared_rows
        ]
        if float(median(residuals)) > threshold:
            current["warnings"] = _normalize_warnings(
                [*current["warnings"], "temporal_disagreement"]
            )

    return ordered


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def build_preannotations(
    candidate_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> Path:
    """Build deterministic suggestions without writing inside the candidate tree."""

    candidate_root = Path(candidate_root)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not candidate_root.is_dir():
        raise FileNotFoundError(candidate_root)
    _validate_output_location(candidate_root, checkpoint_path, output_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    candidates = _load_manifest(candidate_root)
    for candidate in candidates:
        _load_validated_jpeg(candidate)

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    annotator = HoseAnnotator(checkpoint_path, device=device)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        original = _load_validated_jpeg(candidate)
        enhanced = enhance_low_light(original)

        original_model, original_model_warnings = _unpack_predictions(
            annotator.predict(original)
        )
        enhanced_model, enhanced_model_warnings = _unpack_predictions(
            annotator.predict(enhanced.image)
        )
        original_color, original_color_warnings = _unpack_predictions(
            extract_color_anchors(original)
        )
        enhanced_color, enhanced_color_warnings = _unpack_predictions(
            extract_color_anchors(enhanced.image)
        )

        fused_model, model_fusion_warnings = fuse_predictions(
            original_model,
            enhanced_model,
            max_disagreement_px=48.0,
        )
        fused_color, color_fusion_warnings = fuse_predictions(
            original_color,
            enhanced_color,
            max_disagreement_px=64.0,
        )
        fused_combined, combined_fusion_warnings = fuse_predictions(
            fused_model,
            fused_color,
            max_disagreement_px=96.0,
        )
        final_anchors, regularization_warnings = regularize_anchors(fused_combined)

        warnings = _normalize_warnings(
            [
                *original_model_warnings,
                *enhanced_model_warnings,
                *original_color_warnings,
                *enhanced_color_warnings,
                *model_fusion_warnings,
                *color_fusion_warnings,
                *combined_fusion_warnings,
                *regularization_warnings,
            ]
        )
        if len(final_anchors) < 3:
            warnings = _normalize_warnings([*warnings, "insufficient_anchor_count"])

        records.append(
            {
                "stem": candidate.stem,
                "target_timestamp_seconds": candidate.target_timestamp_seconds,
                "image_sha256": candidate.image_sha256,
                "anchors": [_serialize_anchor(anchor) for anchor in final_anchors],
                "warnings": warnings,
                "source_metrics": {
                    "model": {
                        "original_count": len(original_model),
                        "enhanced_count": len(enhanced_model),
                        "fused_count": len(fused_model),
                    },
                    "color": {
                        "original_count": len(original_color),
                        "enhanced_count": len(enhanced_color),
                        "fused_count": len(fused_color),
                    },
                    "combined_fused_count": len(fused_combined),
                    "final_count": len(final_anchors),
                    "enhancement": {
                        "before": _serialize_metrics(enhanced.before),
                        "after": _serialize_metrics(enhanced.after),
                        "params": _serialize_metrics(enhanced.params),
                    },
                },
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "records": add_temporal_warnings(records),
    }
    _write_json_atomic(output_path, document)
    return output_path
