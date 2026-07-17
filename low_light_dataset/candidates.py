"""Extract auditable, event-enriched unlabeled frames from a video."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .annotation import extract_color_anchors
from .artifacts import save_jpeg


@dataclass(frozen=True)
class CandidateMetrics:
    target_timestamp_seconds: float
    actual_timestamp_seconds: float
    frame_difference: float
    optical_flow: float
    brightness_mean: float
    brightness_delta: float
    dark_fraction: float
    laplacian_variance: float
    color_anchor_count: int
    difference_hash_hex: str


@dataclass(frozen=True)
class CandidateRecord:
    stem: str
    source: str
    event_reason: str
    event_score: float
    metrics: CandidateMetrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def _read_at(capture: cv2.VideoCapture, timestamp: float) -> tuple[np.ndarray, float]:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to decode video at {timestamp:.3f}s")
    if frame.shape[:2] != (480, 640):
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    actual = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    return frame, actual


def _metrics(
    frame: np.ndarray,
    timestamp: float,
    actual: float,
    previous_small: np.ndarray | None,
    previous_brightness: float | None,
) -> tuple[CandidateMetrics, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    brightness = float(gray.mean())
    if previous_small is None:
        frame_difference = 0.0
        optical_flow = 0.0
    else:
        frame_difference = float(
            np.mean(np.abs(small.astype(np.float32) - previous_small.astype(np.float32)))
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous_small, small, None, 0.5, 3, 15, 2, 5, 1.1, 0
        )
        optical_flow = float(np.median(np.linalg.norm(flow, axis=2)))
    digest = difference_hash(frame)
    result = CandidateMetrics(
        target_timestamp_seconds=float(timestamp),
        actual_timestamp_seconds=actual,
        frame_difference=frame_difference,
        optical_flow=optical_flow,
        brightness_mean=brightness,
        brightness_delta=0.0 if previous_brightness is None else abs(brightness - previous_brightness),
        dark_fraction=float(np.mean(gray < 48)),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        color_anchor_count=len(extract_color_anchors(frame)),
        difference_hash_hex=f"{digest:016x}",
    )
    return result, small


def _scale(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    return max(float(np.percentile(positive, 90)), 1e-6) if positive else 1.0


def _select_records(metrics: list[CandidateMetrics]) -> list[CandidateRecord]:
    flow_scale = _scale([item.optical_flow for item in metrics])
    diff_scale = _scale([item.frame_difference for item in metrics])
    light_scale = _scale([item.brightness_delta for item in metrics])
    scored: list[tuple[float, str, CandidateMetrics]] = []
    for item in metrics:
        components = {
            "motion": min(item.optical_flow / flow_scale, 2.0) * 0.45,
            "scene_change": min(item.frame_difference / diff_scale, 2.0) * 0.40,
            "lighting_change": min(item.brightness_delta / light_scale, 2.0) * 0.15,
        }
        reason = max(components, key=components.get)
        scored.append((sum(components.values()), reason, item))

    bases = [
        CandidateRecord(
            stem=f"camera_full_rgb_v2_t{int(round(item.target_timestamp_seconds * 1000)):09d}",
            source="base", event_reason="one_second_base", event_score=0.0,
            metrics=item,
        )
        for item in metrics
        if math.isclose(item.target_timestamp_seconds % 1.0, 0.0, abs_tol=1e-7)
    ]
    best_by_second: dict[int, tuple[float, str, CandidateMetrics]] = {}
    for score, reason, item in scored:
        if math.isclose(item.target_timestamp_seconds % 1.0, 0.0, abs_tol=1e-7):
            continue
        second = int(item.target_timestamp_seconds)
        if second not in best_by_second or score > best_by_second[second][0]:
            best_by_second[second] = (score, reason, item)

    selected = list(bases)
    base_hashes = [
        (record.metrics.target_timestamp_seconds,
         int(record.metrics.difference_hash_hex, 16)) for record in bases
    ]
    for second in sorted(best_by_second):
        score, reason, item = best_by_second[second]
        digest = int(item.difference_hash_hex, 16)
        duplicate = any(
            abs(item.target_timestamp_seconds - kept_time) <= 1.0
            and hamming_distance(digest, kept_hash) <= 3
            for kept_time, kept_hash in base_hashes
        )
        if duplicate:
            continue
        selected.append(
            CandidateRecord(
                stem=f"camera_full_rgb_v2_t{int(round(item.target_timestamp_seconds * 1000)):09d}",
                source="event", event_reason=reason, event_score=float(score),
                metrics=item,
            )
        )
    return sorted(selected, key=lambda item: item.metrics.target_timestamp_seconds)


def _write_manifest(path: Path, records: list[CandidateRecord]) -> None:
    fields = [
        "stem", "source", "event_reason", "event_score",
        "target_timestamp_seconds", "actual_timestamp_seconds",
        "frame_difference", "optical_flow", "brightness_mean",
        "brightness_delta", "dark_fraction", "laplacian_variance",
        "color_anchor_count", "difference_hash_hex", "image_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "stem": record.stem,
                "source": record.source,
                "event_reason": record.event_reason,
                "event_score": f"{record.event_score:.6f}",
                **asdict(record.metrics),
                "image_sha256": _sha256(path.parent / "frames" / f"{record.stem}.jpg"),
            }
            for key, value in list(row.items()):
                if isinstance(value, float):
                    row[key] = f"{value:.6f}"
            writer.writerow(row)


def _write_contact_sheets(root: Path, records: list[CandidateRecord]) -> None:
    sheet_dir = root / "contact_sheets"
    sheet_dir.mkdir()
    columns, rows_per_page = 5, 4
    cell_width, image_height, label_height = 320, 240, 32
    per_page = columns * rows_per_page
    for page, start in enumerate(range(0, len(records), per_page), start=1):
        canvas = Image.new(
            "RGB", (cell_width * columns, (image_height + label_height) * rows_per_page),
            (16, 16, 16),
        )
        draw = ImageDraw.Draw(canvas)
        for offset, record in enumerate(records[start:start + per_page]):
            row, column = divmod(offset, columns)
            x = column * cell_width
            y = row * (image_height + label_height)
            with Image.open(root / "frames" / f"{record.stem}.jpg") as image:
                thumb = image.convert("RGB").resize((cell_width, image_height))
            canvas.paste(thumb, (x, y + label_height))
            label = (
                f"{record.metrics.target_timestamp_seconds:.2f}s {record.source} "
                f"flow={record.metrics.optical_flow:.2f} L={record.metrics.brightness_mean:.0f} "
                f"a={record.metrics.color_anchor_count}"
            )
            draw.text((x + 4, y + 8), label, fill=(255, 255, 0))
        canvas.save(sheet_dir / f"candidates_{page:03d}.jpg", quality=90)


def extract_candidate_set(
    video_path: Path,
    output_root: Path,
    duration_seconds: float = 300.0,
    analysis_interval_seconds: float = 0.25,
) -> Path:
    """Analyze and extract base plus event frames without creating labels."""

    video_path = Path(video_path)
    output_root = Path(output_root)
    if duration_seconds <= 0 or analysis_interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        raise FileExistsError(staging)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps if fps > 0 else duration_seconds
    limit = min(float(duration_seconds), float(video_duration))
    timestamps = np.arange(0.0, limit - 1e-9, analysis_interval_seconds).tolist()
    metrics: list[CandidateMetrics] = []
    previous_small = None
    previous_brightness = None
    rejected: list[float] = []
    for timestamp in timestamps:
        try:
            frame, actual = _read_at(capture, timestamp)
        except RuntimeError:
            rejected.append(float(timestamp))
            continue
        item, previous_small = _metrics(
            frame, timestamp, actual, previous_small, previous_brightness
        )
        previous_brightness = item.brightness_mean
        metrics.append(item)
    capture.release()
    records = _select_records(metrics)
    if not records:
        raise RuntimeError("no candidate frames were selected")

    (staging / "frames").mkdir(parents=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index, record in enumerate(records, start=1):
            frame, _ = _read_at(capture, record.metrics.target_timestamp_seconds)
            save_jpeg(staging / "frames" / f"{record.stem}.jpg", frame, quality=95)
            if index % 50 == 0 or index == len(records):
                print(f"candidate extraction {index}/{len(records)}", flush=True)
    finally:
        capture.release()

    _write_manifest(staging / "manifest.csv", records)
    _write_contact_sheets(staging, records)
    numeric_fields = [
        "frame_difference", "optical_flow", "brightness_mean",
        "brightness_delta", "dark_fraction", "laplacian_variance",
    ]
    distributions = {}
    for field in numeric_fields:
        values = np.asarray([getattr(record.metrics, field) for record in records])
        distributions[field] = {
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    summary = {
        "schema_version": 1,
        "video": str(video_path.resolve()),
        "video_sha256": _sha256(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "video_duration_seconds": video_duration,
        "requested_duration_seconds": duration_seconds,
        "analysis_interval_seconds": analysis_interval_seconds,
        "analysis_timestamp_count": len(timestamps),
        "rejected_decode_timestamps": rejected,
        "candidate_count": len(records),
        "base_count": sum(record.source == "base" for record in records),
        "event_count": sum(record.source == "event" for record in records),
        "event_reason_counts": {
            reason: sum(record.event_reason == reason for record in records)
            for reason in sorted({record.event_reason for record in records})
        },
        "distributions": distributions,
    }
    summary_path = staging / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_root)
    return output_root / "summary.json"

