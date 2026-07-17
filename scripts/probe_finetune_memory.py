"""Select the largest safe fine-tuning micro-batch on the current GPU."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model import parsingNet
from train_alian import calc_loss, inference
from utils.factory import get_loss_dict
from utils_alian.dataloader_alian import get_train_loader
from utils_alian.finetune_utils import load_weights_only, select_safe_batch


class NullLogger:
    def add_scalar(self, *args, **kwargs):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidates", type=int, nargs="+", default=[8, 4, 2, 1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-fraction", type=float, default=0.90)
    return parser


def _probe_candidate(source: Path, checkpoint: Path, batch_size: int,
                     safety_fraction: float) -> dict:
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    result = {
        "batch_size": batch_size,
        "trials": [],
        "peak_allocated_mib": [],
        "peak_reserved_mib": [],
        "error": None,
    }
    model = loader = optimizer = scaler = iterator = None
    try:
        model = parsingNet(
            pretrained=False,
            backbone="18",
            cls_dim=(51, 18, 1),
            use_aux=True,
        ).to(device)
        load_weights_only(model, checkpoint)
        loader = get_train_loader(
            batch_size,
            str(source),
            50,
            True,
            False,
            1,
            num_workers=0,
            low_light_exposure=1,
            seed=20260716,
        )
        iterator = iter(loader)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5, weight_decay=1e-5
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        loss_options = Namespace(use_aux=True, sim_loss_w=0, shp_loss_w=0)
        loss_dict = get_loss_dict(loss_options)
        logger = NullLogger()
        model.train()
        for trial in range(2):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                data = next(iterator)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                values = inference(model, data, True, device)
                loss = calc_loss(loss_dict, values, logger, trial)
            if not torch.isfinite(loss):
                raise FloatingPointError("memory probe produced non-finite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            has_margin = peak_reserved <= total_memory * safety_fraction
            result["trials"].append(bool(has_margin))
            result["peak_allocated_mib"].append(
                round(peak_allocated / (1024 * 1024), 2)
            )
            result["peak_reserved_mib"].append(
                round(peak_reserved / (1024 * 1024), 2)
            )
            if not has_margin:
                result["error"] = (
                    f"peak reserved memory exceeds {safety_fraction:.0%} "
                    "safety threshold"
                )
                break
    except torch.cuda.OutOfMemoryError as error:
        result["trials"].append(False)
        result["error"] = f"CUDA OOM: {error}"
    finally:
        del iterator, scaler, optimizer, loader, model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not 0.0 < args.safety_fraction < 1.0:
        raise ValueError("safety fraction must be between zero and one")
    results = []
    for candidate in args.candidates:
        if candidate <= 0 or 64 % candidate != 0:
            raise ValueError(f"candidate must divide effective batch 64: {candidate}")
        print(f"Probing micro-batch {candidate}...", flush=True)
        result = _probe_candidate(
            args.source, args.checkpoint, candidate, args.safety_fraction
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if len(result["trials"]) >= 2 and all(result["trials"][:2]):
            break
    selected = select_safe_batch(results, effective_batch_size=64)
    document = {
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
        ),
        "safety_fraction": args.safety_fraction,
        "candidates": results,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
