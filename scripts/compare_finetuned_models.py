"""Compare the old and fine-tuned models on formal and low-light data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model import parsingNet
from train_alian import inference, resolve_val_data
from utils.factory import get_loss_dict, get_metric_dict
from utils.metrics import reset_metrics, update_metrics
from utils_alian.dataloader_alian import get_val_loader
from utils_alian.finetune_utils import load_weights_only


def _metric_rows(old_metrics, new_metrics):
    rows = []
    keys = sorted(set(old_metrics) & set(new_metrics))
    for key in keys:
        old_value = float(old_metrics[key])
        new_value = float(new_metrics[key])
        rows.append(
            f"| {key} | {old_value:.6f} | {new_value:.6f} | "
            f"{new_value - old_value:+.6f} |"
        )
    return rows


def build_comparison_report(old, new):
    lines = [
        "# 新旧模型对比报告",
        "",
        "## 867组正式测试集",
        "",
        "| 指标 | 旧模型 | 新模型 | 新-旧 |",
        "|---|---:|---:|---:|",
        *_metric_rows(old["test"], new["test"]),
        "",
        "## 60组低照度诊断数据",
        "",
        "| 指标 | 旧模型 | 新模型 | 新-旧 |",
        "|---|---:|---:|---:|",
        *_metric_rows(old["low_light"], new["low_light"]),
        "",
        "> 注意：低照度60组数据已参与训练，不能作为独立泛化指标；",
        "> 该结果只用于诊断新增场景的拟合变化。",
        "",
    ]
    return "\n".join(lines)


class NullLogger:
    def add_scalar(self, *args, **kwargs):
        return None


def _make_model(checkpoint_path, device):
    model = parsingNet(
        pretrained=False,
        backbone="18",
        cls_dim=(51, 18, 1),
        use_aux=True,
    ).to(device)
    metadata = load_weights_only(model, checkpoint_path)
    model.eval()
    return model, metadata


def _build_low_light_loader(train_root, batch_size, num_workers):
    full_loader = get_val_loader(
        batch_size,
        str(train_root),
        50,
        True,
        False,
        1,
        num_workers=num_workers,
    )
    indices = [
        index
        for index, path in enumerate(full_loader.dataset.img_paths)
        if Path(path).stem.startswith("lowlight_camera_full_rgb_")
    ]
    if len(indices) != 60:
        raise ValueError(f"expected 60 low-light samples, got {len(indices)}")
    subset = torch.utils.data.Subset(full_loader.dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def evaluate(model, loader, device):
    options = Namespace(
        use_aux=True,
        sim_loss_w=0,
        shp_loss_w=0,
        griding_num=50,
        num_lanes=1,
    )
    loss_dict = get_loss_dict(options)
    metrics = get_metric_dict(options)
    logger = NullLogger()
    totals = {name: 0.0 for name in metrics["name"]}
    total_loss = 0.0
    batches = 0
    grid_error_sum = 0.0
    valid_points = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            reset_metrics(metrics)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                results = inference(model, data, True, device)
                loss = 0
                for index, operation in enumerate(loss_dict["op"]):
                    values = [
                        results[source]
                        for source in loss_dict["data_src"][index]
                    ]
                    loss += operation(*values) * loss_dict["weight"][index]
            resolved = resolve_val_data(results, True)
            update_metrics(metrics, resolved)
            for name, operation in zip(metrics["name"], metrics["op"]):
                totals[name] += float(operation.get())
            prediction = resolved["cls_out"]
            target = results["cls_label"]
            valid = target != 50
            grid_error_sum += float(
                torch.abs(prediction[valid] - target[valid]).sum().cpu()
            )
            valid_points += int(valid.sum().cpu())
            total_loss += float(loss.detach().cpu())
            batches += 1
    elapsed = time.perf_counter() - started
    output = {"loss": total_loss / batches}
    output.update({name: value / batches for name, value in totals.items()})
    mean_grid_error = grid_error_sum / max(valid_points, 1)
    output.update(
        {
            "mean_grid_error": mean_grid_error,
            "mean_pixel_error": mean_grid_error * 639.0 / 49.0,
            "samples": len(loader.dataset),
            "seconds": elapsed,
            "milliseconds_per_sample": elapsed * 1000.0 / len(loader.dataset),
        }
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    test_loader = get_val_loader(
        args.batch_size,
        str(args.test),
        50,
        True,
        False,
        1,
        num_workers=args.num_workers,
    )
    low_light_loader = _build_low_light_loader(
        args.train, args.batch_size, args.num_workers
    )
    results = {}
    metadata = {}
    for label, checkpoint in (("old", args.old), ("new", args.new)):
        model, model_metadata = _make_model(checkpoint, device)
        results[label] = {
            "test": evaluate(model, test_loader, device),
            "low_light": evaluate(model, low_light_loader, device),
        }
        metadata[label] = model_metadata
        del model
        torch.cuda.empty_cache()
    document = {"metrics": results, "metadata": metadata}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_comparison_report(results["old"], results["new"])
    (args.output / "comparison_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
