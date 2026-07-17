"""Frame sampling and adaptive low-light image restoration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameMetrics:
    mean: float
    median: float
    p05: float
    p95: float
    dark_fraction: float
    highlight_fraction: float
    laplacian_variance: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancementParams:
    gamma: float
    retinex_weight: float
    clahe_clip_limit: float
    denoise_sigma_color: float
    sharpen_amount: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancedFrame:
    image: np.ndarray
    before: FrameMetrics
    after: FrameMetrics
    params: EnhancementParams


def sample_times(
    duration_seconds: int = 300, interval_seconds: int = 5
) -> tuple[int, ...]:
    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    return tuple(range(0, duration_seconds, interval_seconds))


def validate_bgr(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")


def measure_frame(frame_bgr: np.ndarray) -> FrameMetrics:
    validate_bgr(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return FrameMetrics(
        mean=float(gray.mean()),
        median=float(np.median(gray)),
        p05=float(np.percentile(gray, 5)),
        p95=float(np.percentile(gray, 95)),
        dark_fraction=float((gray < 40).mean()),
        highlight_fraction=float((gray > 245).mean()),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def _limited_gray_world(
    frame_bgr: np.ndarray, gain_min: float = 0.85, gain_max: float = 1.18
) -> np.ndarray:
    means = frame_bgr.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1.0), gain_min, gain_max)
    balanced = frame_bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _robust_uint8(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lower, upper = np.percentile(values, (low, high))
    if upper <= lower + 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - lower) * 255.0 / (upper - lower)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _protect_highlights(original_l: np.ndarray, candidate_l: np.ndarray) -> np.ndarray:
    mask = np.clip(
        (original_l.astype(np.float32) - 180.0) / 65.0,
        0.0,
        1.0,
    )
    protected = (
        candidate_l.astype(np.float32) * (1.0 - mask)
        + original_l.astype(np.float32) * mask
    )
    return np.clip(protected, 0, 255).astype(np.uint8)


def _conditional_unsharp(
    frame_bgr: np.ndarray,
    dark_fraction: float,
    laplacian_variance: float,
) -> tuple[np.ndarray, float]:
    amount = float(np.clip(0.45 - 0.35 * dark_fraction, 0.10, 0.45))
    if laplacian_variance > 500.0:
        amount *= 0.5
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(frame_bgr, 1.0 + amount, blurred, -amount, 0)
    return sharpened, amount


def enhance_low_light(frame_bgr: np.ndarray) -> EnhancedFrame:
    """Enhance a frame while adapting strength to the measured illumination."""

    validate_bgr(frame_bgr)
    before = measure_frame(frame_bgr)
    darkness = float(np.clip((115.0 - before.median) / 90.0, 0.0, 1.0))
    retinex_weight = float(0.08 + 0.52 * darkness)
    gamma = float(
        np.clip(
            np.log(115.0 / 255.0)
            / np.log(max(before.median, 1.0) / 255.0),
            0.55,
            1.0,
        )
    )

    balanced = _limited_gray_world(frame_bgr)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    sigma_color = float(24.0 + 20.0 * darkness)
    denoised = cv2.bilateralFilter(
        lightness,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=35.0,
    )
    illumination = cv2.GaussianBlur(denoised, (0, 0), sigmaX=31.0)
    reflectance = np.log1p(denoised.astype(np.float32)) - np.log1p(
        illumination.astype(np.float32)
    )
    retinex = _robust_uint8(reflectance)
    restored = cv2.addWeighted(
        denoised,
        1.0 - retinex_weight,
        retinex,
        retinex_weight,
        0,
    )

    gamma_lut = np.clip(
        np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    toned = cv2.LUT(restored, gamma_lut)
    clahe_clip = float(1.6 + 0.8 * darkness)
    local_contrast = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8),
    ).apply(toned)
    combined = cv2.addWeighted(toned, 0.65, local_contrast, 0.35, 0)
    final_lightness = _protect_highlights(lightness, combined)
    restored_bgr = cv2.cvtColor(
        cv2.merge((final_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    output, sharpen_amount = _conditional_unsharp(
        restored_bgr,
        before.dark_fraction,
        before.laplacian_variance,
    )
    params = EnhancementParams(
        gamma=gamma,
        retinex_weight=retinex_weight,
        clahe_clip_limit=clahe_clip,
        denoise_sigma_color=sigma_color,
        sharpen_amount=sharpen_amount,
    )
    return EnhancedFrame(
        image=output,
        before=before,
        after=measure_frame(output),
        params=params,
    )


def read_video_frame(
    capture: cv2.VideoCapture, timestamp_seconds: int
) -> tuple[np.ndarray, float]:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at {timestamp_seconds}s")
    actual_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    return frame, actual_seconds


def save_jpeg(path: Path, frame_bgr: np.ndarray, quality: int = 95) -> None:
    validate_bgr(frame_bgr)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
    )
