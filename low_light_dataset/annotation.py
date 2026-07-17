"""Model-assisted hose centerline annotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.interpolate
import scipy.special
import torch
from PIL import Image

from model.model import parsingNet


ROW_ANCHORS = np.asarray(
    [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
    dtype=np.int32,
)


@dataclass(frozen=True)
class AnchorPrediction:
    y: int
    x: float
    confidence: float
    source: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def extract_color_anchors(frame_bgr: np.ndarray) -> list[AnchorPrediction]:
    """Find row-anchor centers of the dominant connected red hose region."""

    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be uint8 BGR with three channels")
    height, width = frame_bgr.shape[:2]
    blue, green, red = cv2.split(frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = cv2.split(hsv)

    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    dominance = red_i - np.maximum(green_i, blue_i)
    red_hue = (hue <= 18) | (hue >= 165)
    mask = (
        (dominance >= 18)
        & (red_i >= 55)
        & (saturation >= 45)
        & red_hue
    ).astype(np.uint8)
    mask[: int(height * 0.32)] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return []
    best_label = 0
    best_score = 0.0
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < 120 or component_height < 30:
            continue
        bottom = y + component_height
        score = float(area) * (1.0 + component_height / height) * (1.0 + bottom / height)
        if bottom < int(height * 0.60):
            score *= 0.2
        if component_width > width * 0.75:
            score *= 0.2
        if score > best_score:
            best_score = score
            best_label = label
    if best_label == 0:
        return []

    component = components == best_label
    points: list[AnchorPrediction] = []
    for row_anchor in ROW_ANCHORS:
        y = int(round(float(row_anchor) * (height - 1.0) / 287.0))
        top = max(0, y - 5)
        bottom = min(height, y + 6)
        rows, columns = np.where(component[top:bottom])
        if columns.size < 8:
            continue
        x = float(np.median(columns))
        density = columns.size / max(1.0, (bottom - top) * 80.0)
        confidence = float(np.clip(0.50 + density, 0.50, 0.98))
        points.append(AnchorPrediction(y, x, confidence, "color"))
    return points


def decode_logits(
    logits: np.ndarray,
    width: int,
    height: int,
    min_confidence: float = 0.35,
) -> list[AnchorPrediction]:
    expected_shape = (1, 51, 18, 1)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"expected logits shape {expected_shape}, got {tuple(logits.shape)}")
    if width <= 1 or height <= 1:
        raise ValueError("image dimensions must be greater than one")

    lane_logits = logits[0, :, :, 0]
    full_probabilities = scipy.special.softmax(lane_logits, axis=0)
    foreground_probabilities = scipy.special.softmax(lane_logits[:50], axis=0)
    expected_grid = (
        foreground_probabilities * np.arange(1, 51, dtype=np.float32)[:, None]
    ).sum(axis=0)

    points: list[AnchorPrediction] = []
    for index, row_anchor in enumerate(ROW_ANCHORS):
        confidence = 1.0 - float(full_probabilities[50, index])
        if confidence < min_confidence:
            continue
        x = (float(expected_grid[index]) - 1.0) * (width - 1.0) / 49.0
        y = int(round(float(row_anchor) * (height - 1.0) / 287.0))
        points.append(
            AnchorPrediction(
                y=y,
                x=float(np.clip(x, 0.0, width - 1.0)),
                confidence=confidence,
                source="model",
            )
        )
    return points


def _by_y(points: list[AnchorPrediction]) -> dict[int, AnchorPrediction]:
    result: dict[int, AnchorPrediction] = {}
    for point in points:
        current = result.get(point.y)
        if current is None or point.confidence > current.confidence:
            result[point.y] = point
    return result


def fuse_predictions(
    original: list[AnchorPrediction],
    enhanced: list[AnchorPrediction],
    max_disagreement_px: float = 48.0,
) -> tuple[list[AnchorPrediction], list[str]]:
    original_by_y = _by_y(original)
    enhanced_by_y = _by_y(enhanced)
    fused: list[AnchorPrediction] = []
    warnings: list[str] = []

    for y in sorted(set(original_by_y) | set(enhanced_by_y)):
        left = original_by_y.get(y)
        right = enhanced_by_y.get(y)
        if left is None:
            fused.append(
                AnchorPrediction(y, right.x, right.confidence, "enhanced")  # type: ignore[union-attr]
            )
            continue
        if right is None:
            fused.append(AnchorPrediction(y, left.x, left.confidence, "original"))
            continue

        if abs(left.x - right.x) > max_disagreement_px:
            chosen = right if right.confidence >= left.confidence else left
            fused.append(
                AnchorPrediction(y, chosen.x, chosen.confidence, chosen.source)
            )
            warnings.append(f"prediction_disagreement_y_{y}")
            continue

        total = left.confidence + right.confidence
        x = (left.x * left.confidence + right.x * right.confidence) / max(total, 1e-8)
        confidence = 1.0 - (1.0 - left.confidence) * (1.0 - right.confidence)
        fused.append(AnchorPrediction(y, float(x), float(confidence), "fused"))
    return fused, warnings


def regularize_anchors(
    predictions: list[AnchorPrediction],
    outlier_residual_px: float = 96.0,
) -> tuple[list[AnchorPrediction], list[str]]:
    ordered = sorted(_by_y(predictions).values(), key=lambda item: item.y)
    if len(ordered) < 3:
        return ordered, ["insufficient_anchor_count"]

    remove_indices: set[int] = set()
    warnings: list[str] = []
    for index in range(1, len(ordered) - 1):
        previous = ordered[index - 1]
        current = ordered[index]
        following = ordered[index + 1]
        span = following.y - previous.y
        if span <= 0:
            continue
        fraction = (current.y - previous.y) / span
        expected_x = previous.x + fraction * (following.x - previous.x)
        residual = abs(current.x - expected_x)
        neighbor_span = abs(following.x - previous.x)
        if residual > outlier_residual_px and neighbor_span < outlier_residual_px:
            remove_indices.add(index)
            warnings.append(f"removed_outlier_y_{current.y}")

    cleaned = [point for index, point in enumerate(ordered) if index not in remove_indices]
    if len(cleaned) < 3:
        warnings.append("insufficient_anchor_count_after_regularization")
    return cleaned, warnings


def rasterize_centerline(
    anchors: list[AnchorPrediction],
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    ordered = sorted(_by_y(anchors).values(), key=lambda item: item.y)
    if len(ordered) < 3:
        raise ValueError("at least three valid anchors are required")

    ys = np.asarray([point.y for point in ordered], dtype=np.float64)
    xs = np.asarray([point.x for point in ordered], dtype=np.float64)
    if np.any(np.diff(ys) <= 0):
        raise ValueError("anchor y coordinates must be strictly increasing")

    first_y = max(0, int(np.ceil(ys[0])))
    last_y = min(height - 1, int(np.floor(ys[-1])))
    if first_y > last_y:
        raise ValueError("anchors are outside the output image")
    dense_y = np.arange(first_y, last_y + 1, dtype=np.int32)
    interpolator = scipy.interpolate.PchipInterpolator(ys, xs, extrapolate=False)
    dense_x = np.clip(np.rint(interpolator(dense_y)), 0, width - 1).astype(np.int32)

    label = np.zeros((height, width), dtype=np.uint8)
    label[dense_y, dense_x] = 1
    return label


class HoseAnnotator:
    """Load the delivered lane-style model and return hose anchor predictions."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if device != "cpu" and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.model = parsingNet(
            pretrained=False,
            cls_dim=(51, 18, 1),
            use_aux=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" not in checkpoint:
            raise KeyError("checkpoint does not contain a 'model' state dictionary")
        state = {
            key.removeprefix("module."): value
            for key, value in checkpoint["model"].items()
        }
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("aux_")]
        unexpected = [
            key for key in incompatible.unexpected_keys if not key.startswith("aux_")
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"incompatible checkpoint; missing={missing}, unexpected={unexpected}"
            )
        self.model.to(self.device).eval()

    def predict(self, frame_bgr: np.ndarray) -> list[AnchorPrediction]:
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be uint8 BGR with three channels")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(
            (384, 288),
            resample=Image.Resampling.BILINEAR,
        ).convert("L")
        gray = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(gray)[None, None].repeat(1, 3, 1, 1)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[
            None, :, None, None
        ]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[
            None, :, None, None
        ]
        normalized = ((tensor - mean) / std).to(self.device)
        with torch.no_grad():
            logits = self.model(normalized).detach().cpu().numpy()
        return decode_logits(
            logits,
            width=frame_bgr.shape[1],
            height=frame_bgr.shape[0],
        )
