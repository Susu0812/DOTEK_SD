"""Train or fine-tune the custom Ultra-Fast hose detector."""

from __future__ import annotations

import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.model import parsingNet
from utils.common import (
    cp_projects,
    get_logger,
    save_training_checkpoint,
)
from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import (
    get_loss_dict,
    get_metric_dict,
    get_optimizer,
    get_scheduler,
)
from utils.metrics import reset_metrics, update_metrics
from utils_alian.config import (
    get_args,
    get_work_dir,
    validate_finetune_options,
)
from utils_alian.dataloader_alian import get_train_loader, get_val_loader
from utils_alian.finetune_utils import (
    AccumulationStepper,
    TrainingProgressRecorder,
    load_weights_only,
    optimizer_updates_per_epoch,
    sha256_file,
)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inference(net, data_label, use_aux, device):
    if use_aux:
        img, cls_label, seg_label = data_label
        img = img.to(device, non_blocking=True)
        cls_label = cls_label.long().to(device, non_blocking=True)
        seg_label = seg_label.long().to(device, non_blocking=True)
        cls_out, seg_out = net(img)
        return {
            "cls_out": cls_out,
            "cls_label": cls_label,
            "seg_out": seg_out,
            "seg_label": seg_label,
        }
    img, cls_label = data_label
    img = img.to(device, non_blocking=True)
    cls_label = cls_label.long().to(device, non_blocking=True)
    return {"cls_out": net(img), "cls_label": cls_label}


def resolve_val_data(results, use_aux):
    resolved = dict(results)
    resolved["cls_out"] = torch.argmax(results["cls_out"], dim=1)
    if use_aux:
        resolved["seg_out"] = torch.argmax(results["seg_out"], dim=1)
    return resolved


def calc_loss(loss_dict, results, logger, global_step):
    total = 0
    for index, name in enumerate(loss_dict["name"]):
        values = [results[source] for source in loss_dict["data_src"][index]]
        current = loss_dict["op"][index](*values)
        if global_step % 20 == 0:
            logger.add_scalar(f"loss/{name}", current, global_step)
        total += current * loss_dict["weight"][index]
    return total


def _metric_values(metric_dict):
    return {
        name: float(operation.get())
        for name, operation in zip(metric_dict["name"], metric_dict["op"])
    }


def _append_epoch_summary(work_dir, phase, epoch, metrics):
    parts = [f"Epoch[{epoch}]-{phase}"]
    parts.extend(f"avg_{name}={value:.6f}" for name, value in metrics.items())
    with (Path(work_dir) / "train_epoch_summary.txt").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(" ".join(parts) + "\n")
        handle.flush()


def train(
    net,
    data_loader,
    loss_dict,
    optimizer,
    scheduler,
    scaler,
    logger,
    epoch,
    metric_dict,
    options,
    device,
    global_optimizer_step,
    work_dir,
):
    net.train()
    if hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    progress_bar = dist_tqdm(data_loader)
    total_loss = 0.0
    total_batches = 0
    metric_totals = {name: 0.0 for name in metric_dict["name"]}
    stepper = AccumulationStepper(
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        accumulation_steps=options.accumulation_steps,
        global_optimizer_step=global_optimizer_step,
    )

    for batch_index, data_label in enumerate(progress_bar):
        reset_metrics(metric_dict)
        log_step = epoch * len(data_loader) + batch_index
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=options.amp
        ):
            results = inference(net, data_label, options.use_aux, device)
            loss = calc_loss(loss_dict, results, logger, log_step)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at epoch={epoch} batch={batch_index}"
            )
        stepper.backward_and_maybe_step(
            loss,
            batch_index=batch_index,
            total_batches=len(data_loader),
        )

        resolved = resolve_val_data(results, options.use_aux)
        update_metrics(metric_dict, resolved)
        batch_metrics = _metric_values(metric_dict)
        total_loss += float(loss.detach())
        total_batches += 1
        for name, value in batch_metrics.items():
            metric_totals[name] += value
        progress_bar.set_postfix(
            loss=f"{float(loss.detach()):.3f}",
            **{name: f"{value:.3f}" for name, value in batch_metrics.items()},
        )
        if log_step % 20 == 0:
            for name, value in batch_metrics.items():
                logger.add_scalar(f"metric/{name}", value, log_step)
        logger.add_scalar(
            "meta/lr", optimizer.param_groups[0]["lr"], log_step
        )

    metrics = {"loss": total_loss / total_batches}
    metrics.update(
        {name: total / total_batches for name, total in metric_totals.items()}
    )
    logger.add_scalar("epoch/loss", metrics["loss"], epoch)
    for name in metric_dict["name"]:
        logger.add_scalar(f"epoch/{name}", metrics[name], epoch)
    _append_epoch_summary(work_dir, "train", epoch, metrics)
    return metrics, stepper.global_optimizer_step


def val(
    net,
    data_loader,
    loss_dict,
    logger,
    epoch,
    metric_dict,
    options,
    device,
    work_dir,
):
    net.eval()
    progress_bar = dist_tqdm(data_loader)
    total_loss = 0.0
    total_batches = 0
    metric_totals = {name: 0.0 for name in metric_dict["name"]}
    with torch.no_grad():
        for batch_index, data_label in enumerate(progress_bar):
            reset_metrics(metric_dict)
            log_step = epoch * len(data_loader) + batch_index
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=options.amp
            ):
                results = inference(net, data_label, options.use_aux, device)
                loss = calc_loss(loss_dict, results, logger, log_step)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite validation loss at epoch={epoch} batch={batch_index}"
                )
            resolved = resolve_val_data(results, options.use_aux)
            update_metrics(metric_dict, resolved)
            batch_metrics = _metric_values(metric_dict)
            total_loss += float(loss.detach())
            total_batches += 1
            for name, value in batch_metrics.items():
                metric_totals[name] += value
            progress_bar.set_postfix(
                loss=f"{float(loss.detach()):.3f}",
                **{
                    name: f"{value:.3f}"
                    for name, value in batch_metrics.items()
                },
            )

    metrics = {"loss": total_loss / total_batches}
    metrics.update(
        {name: total / total_batches for name, total in metric_totals.items()}
    )
    logger.add_scalar("val/avg_loss", metrics["loss"], epoch)
    for name in metric_dict["name"]:
        logger.add_scalar(f"val/{name}", metrics[name], epoch)
    _append_epoch_summary(work_dir, "val", epoch, metrics)
    dist_print(f"Epoch {epoch} validation: avg_loss={metrics['loss']:.6f}")
    return metrics


def _write_json_atomic(path, document):
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def main():
    torch.backends.cudnn.benchmark = True
    options = validate_finetune_options(get_args())
    set_reproducible_seed(options.seed)
    device = torch.device("cuda:0")
    work_dir = Path(get_work_dir(options))
    work_dir.mkdir(parents=True, exist_ok=False)
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        raise ValueError("this fine-tuning workflow supports one GPU only")

    dist_print(
        datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")
        + " start training..."
    )
    dist_print(options)
    assert options.backbone in [
        "18", "34", "50", "101", "152", "50next", "101next",
        "50wide", "101wide",
    ]

    train_loader = get_train_loader(
        options.batch_size,
        options.source,
        options.griding_num,
        options.use_aux,
        distributed,
        options.num_lanes,
        num_workers=options.num_workers,
        low_light_exposure=options.low_light_exposure,
        seed=options.seed,
    )
    val_loader = get_val_loader(
        options.batch_size,
        options.val_source,
        options.griding_num,
        options.use_aux,
        distributed,
        options.num_lanes,
        num_workers=options.num_workers,
    )

    pretrained = options.finetune is None and options.resume is None
    net = parsingNet(
        pretrained=pretrained,
        backbone=options.backbone,
        cls_dim=(
            options.griding_num + 1,
            options.row_anchor,
            options.num_lanes,
        ),
        use_aux=options.use_aux,
    ).to(device)

    source_metadata = {}
    source_hash = None
    if options.finetune is not None:
        source_hash = sha256_file(options.finetune)
        source_metadata = load_weights_only(net, options.finetune)
        dist_print(
            "Loaded model weights only from " + options.finetune
            + "; old optimizer state was NOT restored."
        )

    optimizer = get_optimizer(net, options)
    resume_epoch = 0
    best_val_loss = float("inf")
    best_epoch = None
    global_optimizer_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=options.amp)
    if options.resume is not None:
        checkpoint = torch.load(options.resume, map_location="cpu")
        net.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        resume_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        global_optimizer_step = int(
            checkpoint.get("global_optimizer_step", 0)
        )

    updates_per_epoch = optimizer_updates_per_epoch(
        len(train_loader), options.accumulation_steps
    )
    scheduler = get_scheduler(optimizer, options, updates_per_epoch)
    metric_dict = get_metric_dict(options)
    loss_dict = get_loss_dict(options)
    logger = get_logger(str(work_dir), options)
    cp_projects(options.auto_backup, str(work_dir))
    progress_recorder = TrainingProgressRecorder(
        work_dir / "training_progress.jsonl", options.epoch
    )
    configuration = vars(options).copy()
    configuration.update(
        {
            "effective_batch_size": (
                options.batch_size * options.accumulation_steps
            ),
            "optimizer_updates_per_epoch": updates_per_epoch,
            "sampled_indices_per_epoch": len(train_loader.sampler),
            "finetune_source_sha256": source_hash,
            "finetune_source_metadata": source_metadata,
        }
    )
    with (work_dir / "cfg.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps(configuration, ensure_ascii=False, indent=2))

    started = time.time()
    for epoch in range(resume_epoch, options.epoch):
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        dist_print(f"==> Epoch[{epoch}]: Training...")
        train_metrics, global_optimizer_step = train(
            net, train_loader, loss_dict, optimizer, scheduler, scaler,
            logger, epoch, metric_dict, options, device,
            global_optimizer_step, work_dir,
        )
        dist_print("Validating...")
        val_metrics = val(
            net, val_loader, loss_dict, logger, epoch, metric_dict,
            options, device, work_dir,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_training_checkpoint(
                work_dir / "best_model.pth",
                model=net,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                val_loss=val_metrics["loss"],
                best_val_loss=best_val_loss,
                global_optimizer_step=global_optimizer_step,
                options=configuration,
            )
            dist_print(
                f"Best validation loss updated: {best_val_loss:.6f}; "
                f"saved to {work_dir / 'best_model.pth'}"
            )
        save_training_checkpoint(
            work_dir / "latest_model.pth",
            model=net,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            val_loss=val_metrics["loss"],
            best_val_loss=best_val_loss,
            global_optimizer_step=global_optimizer_step,
            options=configuration,
        )
        epoch_seconds = time.time() - epoch_started
        record = progress_recorder.record(
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch_seconds=epoch_seconds,
            elapsed_seconds=time.time() - started,
            learning_rate=optimizer.param_groups[0]["lr"],
            peak_memory_mib=torch.cuda.max_memory_reserved(device)
            / (1024 * 1024),
        )
        dist_print("PROGRESS " + json.dumps(record, ensure_ascii=False))

    result = {
        "status": "complete",
        "work_dir": str(work_dir.resolve()),
        "total_seconds": time.time() - started,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "completed_epochs": options.epoch,
        "global_optimizer_step": global_optimizer_step,
        "options": configuration,
    }
    _write_json_atomic(work_dir / "training_result.json", result)
    logger.close()
    dist_print("TRAINING_COMPLETE " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
