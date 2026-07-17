"""Generate, review, and validate low-light dataset artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .annotation import (
    AnchorPrediction,
    HoseAnnotator,
    extract_color_anchors,
    fuse_predictions,
    rasterize_centerline,
    regularize_anchors,
)
from .image_ops import (
    EnhancedFrame,
    enhance_low_light,
    read_video_frame,
    sample_times,
    save_jpeg,
)


ARTIFACT_DIRS = ("original", "enhanced", "label_initial", "label", "overlay")


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_label(path: Path, label: np.ndarray) -> None:
    if not isinstance(label, np.ndarray) or label.dtype != np.uint8:
        raise TypeError("label must be a uint8 numpy array")
    if label.ndim != 2:
        raise ValueError("label must be single-channel")
    unique = set(int(value) for value in np.unique(label))
    if not unique.issubset({0, 1}):
        raise ValueError(f"label must be binary 0/1, got {sorted(unique)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label, mode="L").save(path, format="PNG", optimize=True)


def save_overlay(path: Path, frame_bgr: np.ndarray, label: np.ndarray) -> None:
    if label.shape != frame_bgr.shape[:2]:
        raise ValueError("overlay label and image dimensions differ")
    visible_line = cv2.dilate(
        (label > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    overlay = frame_bgr.copy()
    overlay[visible_line] = (0, 0, 255)
    save_jpeg(path, overlay, quality=95)


def _anchors_to_dict(points: list[AnchorPrediction]) -> list[dict[str, Any]]:
    return [point.to_dict() for point in points]


def _anchors_from_dict(items: list[dict[str, Any]]) -> list[AnchorPrediction]:
    return [
        AnchorPrediction(
            y=int(item["y"]),
            x=float(item["x"]),
            confidence=float(item.get("confidence", 1.0)),
            source=str(item.get("source", "reviewed")),
        )
        for item in items
    ]


def _build_record(
    stem: str,
    timestamp: int,
    actual_timestamp: float,
    enhancement: EnhancedFrame,
    raw_model: list[AnchorPrediction],
    enhanced_model: list[AnchorPrediction],
    raw_color: list[AnchorPrediction],
    enhanced_color: list[AnchorPrediction],
    final_points: list[AnchorPrediction],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "stem": stem,
        "target_timestamp_seconds": timestamp,
        "actual_timestamp_seconds": actual_timestamp,
        "before": enhancement.before.to_dict(),
        "after": enhancement.after.to_dict(),
        "enhancement": enhancement.params.to_dict(),
        "raw_model_anchors": _anchors_to_dict(raw_model),
        "enhanced_model_anchors": _anchors_to_dict(enhanced_model),
        "raw_color_anchors": _anchors_to_dict(raw_color),
        "enhanced_color_anchors": _anchors_to_dict(enhanced_color),
        "initial_anchors": _anchors_to_dict(final_points),
        "final_anchors": _anchors_to_dict(final_points),
        "warnings": sorted(set(warnings)),
        "reviewed": False,
        "approved": False,
        "hose_visible": None,
        "corrected": False,
    }


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "stem",
        "target_timestamp_seconds",
        "actual_timestamp_seconds",
        "before_mean",
        "before_median",
        "before_dark_fraction",
        "after_mean",
        "after_median",
        "after_dark_fraction",
        "after_highlight_fraction",
        "gamma",
        "retinex_weight",
        "initial_anchor_count",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "stem": record["stem"],
                    "target_timestamp_seconds": record["target_timestamp_seconds"],
                    "actual_timestamp_seconds": f"{record['actual_timestamp_seconds']:.6f}",
                    "before_mean": f"{record['before']['mean']:.4f}",
                    "before_median": f"{record['before']['median']:.4f}",
                    "before_dark_fraction": f"{record['before']['dark_fraction']:.6f}",
                    "after_mean": f"{record['after']['mean']:.4f}",
                    "after_median": f"{record['after']['median']:.4f}",
                    "after_dark_fraction": f"{record['after']['dark_fraction']:.6f}",
                    "after_highlight_fraction": f"{record['after']['highlight_fraction']:.6f}",
                    "gamma": f"{record['enhancement']['gamma']:.6f}",
                    "retinex_weight": f"{record['enhancement']['retinex_weight']:.6f}",
                    "initial_anchor_count": len(record["initial_anchors"]),
                    "warnings": "|".join(record["warnings"]),
                }
            )


def _open_thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _write_contact_sheets(root: Path, records: list[dict[str, Any]]) -> None:
    for old in list(root.glob("preview_*.jpg")) + list(root.glob("overlay_preview_*.jpg")):
        old.unlink()
    per_page = 12
    cell_width, cell_height = 320, 270
    label_height = 22
    for page, start in enumerate(range(0, len(records), per_page), start=1):
        page_records = records[start : start + per_page]
        comparison = Image.new(
            "RGB",
            (cell_width * 2, (cell_height + label_height) * len(page_records)),
            (15, 15, 15),
        )
        overlays = Image.new(
            "RGB",
            (cell_width * 2, (cell_height + label_height) * 6),
            (15, 15, 15),
        )
        comparison_draw = ImageDraw.Draw(comparison)
        overlay_draw = ImageDraw.Draw(overlays)
        for row, record in enumerate(page_records):
            stem = record["stem"]
            y = row * (cell_height + label_height)
            comparison.paste(_open_thumbnail(root / "original" / f"{stem}.jpg", (cell_width, cell_height)), (0, y + label_height))
            comparison.paste(_open_thumbnail(root / "enhanced" / f"{stem}.jpg", (cell_width, cell_height)), (cell_width, y + label_height))
            comparison_draw.text((6, y + 3), f"{record['target_timestamp_seconds']}s original", fill=(255, 255, 0))
            comparison_draw.text((cell_width + 6, y + 3), f"{record['target_timestamp_seconds']}s enhanced", fill=(255, 255, 0))

            column = row % 2
            overlay_row = row // 2
            ox = column * cell_width
            oy = overlay_row * (cell_height + label_height)
            overlays.paste(_open_thumbnail(root / "overlay" / f"{stem}.jpg", (cell_width, cell_height)), (ox, oy + label_height))
            overlay_draw.text((ox + 6, oy + 3), f"{record['target_timestamp_seconds']}s anchors={len(record['final_anchors'])}", fill=(255, 255, 0))
        comparison.save(root / f"preview_{page:02d}.jpg", quality=92)
        overlays.save(root / f"overlay_preview_{page:02d}.jpg", quality=92)


def _create_staging_root(output_root: Path) -> Path:
    staging = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite existing dataset artifacts")
    for directory in ARTIFACT_DIRS:
        (staging / directory).mkdir(parents=True, exist_ok=True)
    return staging


def prepare_dataset(
    video_path: Path,
    output_root: Path,
    checkpoint_path: Path,
    device: str = "cuda",
) -> Path:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count / fps < 300.0:
        capture.release()
        raise ValueError("video is shorter than five minutes")

    staging = _create_staging_root(output_root)
    annotator = HoseAnnotator(checkpoint_path, device=device)
    records: list[dict[str, Any]] = []
    try:
        for timestamp in sample_times():
            print(f"prepare {timestamp:03d}s / 295s", flush=True)
            raw, actual_timestamp = read_video_frame(capture, timestamp)
            enhancement = enhance_low_light(raw)
            stem = f"lowlight_camera_full_rgb_t{timestamp:06d}"
            save_jpeg(staging / "original" / f"{stem}.jpg", raw)
            save_jpeg(staging / "enhanced" / f"{stem}.jpg", enhancement.image)

            raw_model = annotator.predict(raw)
            enhanced_model = annotator.predict(enhancement.image)
            raw_color = extract_color_anchors(raw)
            enhanced_color = extract_color_anchors(enhancement.image)
            model_fused, model_warnings = fuse_predictions(raw_model, enhanced_model)
            color_fused, color_warnings = fuse_predictions(
                raw_color, enhanced_color, max_disagreement_px=64.0
            )
            combined, source_warnings = fuse_predictions(
                model_fused, color_fused, max_disagreement_px=96.0
            )
            final_points, regularization_warnings = regularize_anchors(combined)
            warnings = (
                model_warnings
                + color_warnings
                + source_warnings
                + regularization_warnings
            )
            if len(final_points) >= 3:
                initial_label = rasterize_centerline(final_points)
            else:
                initial_label = np.zeros((480, 640), dtype=np.uint8)
                warnings.append("manual_annotation_required")
            save_label(staging / "label_initial" / f"{stem}.png", initial_label)
            save_overlay(staging / "overlay" / f"{stem}.jpg", enhancement.image, initial_label)
            records.append(
                _build_record(
                    stem,
                    timestamp,
                    actual_timestamp,
                    enhancement,
                    raw_model,
                    enhanced_model,
                    raw_color,
                    enhanced_color,
                    final_points,
                    warnings,
                )
            )
    except Exception:
        capture.release()
        raise
    capture.release()

    _write_manifest(staging / "manifest.csv", records)
    write_json_atomic(
        staging / "annotation.json",
        {"schema_version": 1, "frames": records},
    )
    _write_contact_sheets(staging, records)
    if len(records) != 60:
        raise RuntimeError(f"expected 60 prepared frames, got {len(records)}")
    staging.replace(output_root)
    return output_root


def _load_review_decisions(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("review file must contain a 'decisions' object")
    return decisions


def finalize_labels(
    output_root: Path,
    review_path: Path,
    expected_count: int = 60,
) -> ValidationReport:
    annotation_path = output_root / "annotation.json"
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = document.get("frames", [])
    decisions = _load_review_decisions(review_path)
    expected_stems = {record["stem"] for record in records}
    if set(decisions) != expected_stems:
        missing = sorted(expected_stems - set(decisions))
        extra = sorted(set(decisions) - expected_stems)
        raise ValueError(f"review must cover every frame; missing={missing}, extra={extra}")

    for record in records:
        stem = record["stem"]
        decision = decisions[stem]
        if decision.get("approved") is not True:
            raise ValueError(f"frame is not approved: {stem}")
        hose_visible = decision.get("hose_visible")
        if not isinstance(hose_visible, bool):
            raise ValueError(f"hose_visible must be explicit for {stem}")

        if hose_visible:
            point_items = decision.get("anchors", record["initial_anchors"])
            points, warnings = regularize_anchors(_anchors_from_dict(point_items))
            if len(points) < 3:
                raise ValueError(f"visible hose requires at least three anchors: {stem}")
            label = rasterize_centerline(points)
            record["final_anchors"] = _anchors_to_dict(points)
            record["warnings"] = sorted(set(record["warnings"] + warnings))
            record["corrected"] = "anchors" in decision
        else:
            label = np.zeros((480, 640), dtype=np.uint8)
            record["final_anchors"] = []
            record["corrected"] = bool(record["initial_anchors"])

        record["reviewed"] = True
        record["approved"] = True
        record["hose_visible"] = hose_visible
        save_label(output_root / "label" / f"{stem}.png", label)
        image = load_bgr(output_root / "enhanced" / f"{stem}.jpg")
        save_overlay(output_root / "overlay" / f"{stem}.jpg", image, label)

    write_json_atomic(annotation_path, document)
    _write_contact_sheets(output_root, records)
    return validate_prepared_bundle(output_root, expected_count=expected_count)


def validate_prepared_bundle(
    output_root: Path,
    expected_count: int = 60,
    require_review: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    annotation_path = output_root / "annotation.json"
    if not annotation_path.is_file():
        return ValidationReport(False, ["missing annotation.json"], {})
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = document.get("frames", [])
    except (json.JSONDecodeError, OSError) as error:
        return ValidationReport(False, [f"invalid annotation.json: {error}"], {})
    record_by_stem = {record.get("stem"): record for record in records}

    images = sorted((output_root / "enhanced").glob("*.jpg"))
    labels = sorted((output_root / "label").glob("*.png"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if len(images) != expected_count:
        errors.append(f"expected {expected_count} enhanced images, got {len(images)}")
    if len(labels) != expected_count:
        errors.append(f"expected {expected_count} final labels, got {len(labels)}")
    if len(records) != expected_count:
        errors.append(f"expected {expected_count} annotation records, got {len(records)}")
    if image_stems != label_stems:
        errors.append("image and label stems do not match")
    if image_stems != set(record_by_stem):
        errors.append("artifact stems and annotation stems do not match")

    for path in images:
        try:
            with Image.open(path) as image:
                if image.size != (640, 480) or image.mode != "RGB":
                    errors.append(f"invalid training image format: {path.name}")
        except OSError as error:
            errors.append(f"cannot read image {path.name}: {error}")

    for path in labels:
        record = record_by_stem.get(path.stem, {})
        try:
            with Image.open(path) as image:
                array = np.asarray(image)
                if image.size != (640, 480) or image.mode != "L":
                    errors.append(f"invalid label format: {path.name}")
                unique = set(int(value) for value in np.unique(array))
                if not unique.issubset({0, 1}):
                    errors.append(f"non-binary label: {path.name}")
                if np.any((array > 0).sum(axis=1) > 1):
                    errors.append(
                        f"label has more than one point in a row: {path.name}"
                    )
                positive = int((array > 0).sum())
        except OSError as error:
            errors.append(f"cannot read label {path.name}: {error}")
            continue

        if require_review:
            if record.get("reviewed") is not True or record.get("approved") is not True:
                errors.append(f"unreviewed frame: {path.stem}")
            visible = record.get("hose_visible")
            if visible is True and positive == 0:
                errors.append(f"visible hose has empty label: {path.stem}")
            if visible is False and positive != 0:
                errors.append(f"negative sample has non-empty label: {path.stem}")
            if not isinstance(visible, bool):
                errors.append(f"hose visibility not reviewed: {path.stem}")

    details = {
        "enhanced_images": len(images),
        "labels": len(labels),
        "annotation_records": len(records),
        "expected_count": expected_count,
    }
    return ValidationReport(not errors, errors, details)
