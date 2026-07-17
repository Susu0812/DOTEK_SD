"""Utilities shared by low-light fine-tuning, monitoring, and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


class LowLightExposureSampler(torch.utils.data.Sampler[int]):
    """Shuffle all samples while repeating reviewed low-light samples."""

    def __init__(
        self,
        image_paths: Sequence[str],
        prefix: str,
        exposure: int,
        seed: int,
        expected_low_light_count: int = 60,
    ) -> None:
        if exposure < 1:
            raise ValueError("exposure must be at least one")
        if not prefix:
            raise ValueError("low-light filename prefix must not be empty")
        self.base_indices = list(range(len(image_paths)))
        self.low_light_indices = [
            index
            for index, path in enumerate(image_paths)
            if Path(path).stem.startswith(prefix)
        ]
        if len(self.low_light_indices) != expected_low_light_count:
            raise ValueError(
                f"expected {expected_low_light_count} low-light samples, "
                f"got {len(self.low_light_indices)}"
            )
        self.exposure = int(exposure)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        indices = self.base_indices + self.low_light_indices * (
            self.exposure - 1
        )
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return iter(indices[position] for position in order)

    def __len__(self) -> int:
        return len(self.base_indices) + len(self.low_light_indices) * (
            self.exposure - 1
        )


def optimizer_updates_per_epoch(
    loader_batches: int, accumulation_steps: int
) -> int:
    if loader_batches <= 0 or accumulation_steps <= 0:
        raise ValueError("loader batches and accumulation steps must be positive")
    return int(math.ceil(loader_batches / accumulation_steps))


def select_safe_batch(
    probe_results: Sequence[dict[str, Any]],
    effective_batch_size: int = 64,
) -> dict[str, Any]:
    for result in probe_results:
        batch_size = int(result["batch_size"])
        trials = list(result.get("trials", []))
        if (
            len(trials) >= 2
            and all(trials[:2])
            and effective_batch_size % batch_size == 0
        ):
            selected = dict(result)
            selected["accumulation_steps"] = (
                effective_batch_size // batch_size
            )
            selected["effective_batch_size"] = effective_batch_size
            return selected
    raise RuntimeError("no safe micro-batch passed two complete trials")


class AccumulationStepper:
    """Own gradient accumulation and synchronize scheduler with real updates."""

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: Any,
        accumulation_steps: int,
        global_optimizer_step: int = 0,
    ) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation steps must be positive")
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.accumulation_steps = int(accumulation_steps)
        self.global_optimizer_step = int(global_optimizer_step)
        self.optimizer.zero_grad(set_to_none=True)

    def backward_and_maybe_step(
        self,
        loss: torch.Tensor,
        *,
        batch_index: int,
        total_batches: int,
    ) -> bool:
        if total_batches <= 0 or not 0 <= batch_index < total_batches:
            raise ValueError("invalid batch index/total")
        self.scaler.scale(loss / self.accumulation_steps).backward()
        update_due = (batch_index + 1) % self.accumulation_steps == 0
        final_batch = batch_index + 1 == total_batches
        if not (update_due or final_batch):
            return False
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step(self.global_optimizer_step)
        self.global_optimizer_step += 1
        return True


def _normalise_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_weights_only(
    model: torch.nn.Module, checkpoint_path: Path | str
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {path}")
    state_dict = _normalise_state_dict(checkpoint["model"])
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        key
        for key in model_state
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    ]
    if mismatches:
        raise ValueError(f"checkpoint shape mismatch: {mismatches}")
    model.load_state_dict(state_dict, strict=True)
    return {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model", "optimizer", "scaler"}
    }


def estimate_remaining_seconds(
    epoch_seconds: Sequence[float],
    completed_epochs: int,
    total_epochs: int,
) -> float:
    if completed_epochs < 0 or total_epochs < completed_epochs:
        raise ValueError("invalid completed/total epoch values")
    if not epoch_seconds:
        return 0.0
    recent = [float(value) for value in epoch_seconds[-3:]]
    return sum(recent) / len(recent) * (total_epochs - completed_epochs)


class TrainingProgressRecorder:
    def __init__(self, path: Path | str, total_epochs: int) -> None:
        if total_epochs <= 0:
            raise ValueError("total epochs must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total_epochs = int(total_epochs)
        self.epoch_seconds: list[float] = []

    def record(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        epoch_seconds: float,
        elapsed_seconds: float,
        learning_rate: float,
        peak_memory_mib: float,
    ) -> dict[str, Any]:
        self.epoch_seconds.append(float(epoch_seconds))
        completed = int(epoch)
        record = {
            "epoch": completed,
            "total_epochs": self.total_epochs,
            "remaining_epochs": self.total_epochs - completed,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_seconds": float(epoch_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "eta_seconds": estimate_remaining_seconds(
                self.epoch_seconds, completed, self.total_epochs
            ),
            "learning_rate": float(learning_rate),
            "peak_memory_mib": float(peak_memory_mib),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        return record


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
