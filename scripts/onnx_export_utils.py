from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch


AUXILIARY_PREFIXES = (
    "aux_header2.",
    "aux_header3.",
    "aux_header4.",
    "aux_combine.",
)


def resolve_output_paths(output_dir: Path) -> Tuple[Path, Path]:
    """Return both conventional ONNX names inside one training run directory."""
    output_dir = Path(output_dir).resolve()
    return (
        output_dir / "resnet_288_384.onnx",
        output_dir / "resnet_288_384_best.onnx",
    )


def prepare_inference_state(
    checkpoint_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Validate and remove training-only auxiliary weights for ONNX inference."""
    normalized = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint_state.items()
    }
    unexpected = sorted(
        key
        for key in normalized
        if key not in target_state and not key.startswith(AUXILIARY_PREFIXES)
    )
    if unexpected:
        raise ValueError(f"unexpected non-auxiliary checkpoint keys: {unexpected[:5]}")

    missing = sorted(key for key in target_state if key not in normalized)
    if missing:
        raise ValueError(f"missing inference checkpoint keys: {missing[:5]}")

    mismatched = sorted(
        key
        for key, target in target_state.items()
        if tuple(normalized[key].shape) != tuple(target.shape)
    )
    if mismatched:
        details = [
            f"{key}: checkpoint={tuple(normalized[key].shape)}, target={tuple(target_state[key].shape)}"
            for key in mismatched[:5]
        ]
        raise ValueError(f"shape mismatch for inference keys: {details}")

    return {key: normalized[key] for key in target_state}
