# Low-Light Fine-Tuning and Model Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the existing best hose-recognition model for 50 epochs on the 4918-pair formal dataset, safely fit training into a 2 GB GTX 1050, emphasize the 60 restored low-light samples, compare against the old model, and export verified PyTorch and ONNX deliverables.

**Architecture:** Keep the existing UFLD network, losses, transforms, validation split, and log naming convention. Add isolated utilities for deterministic low-light resampling, gradient accumulation/AMP state, progress recording, memory probing, comparison evaluation, and parameterized ONNX export so the original model and artifacts remain untouched.

**Tech Stack:** Python 3.10, PyTorch 2.1.1 CUDA 11.8, torchvision 0.16, NumPy 1.24.4, Pillow 10.4, ONNX, ONNX Runtime, TensorBoard, built-in `unittest`, JSON/JSONL/SHA-256.

## Global Constraints

- Initialize from `logs/0209_1509_lr_1e-04_b_64/best_model.pth` weights only; never restore its optimizer state.
- Train on `datasets/newdata/train` (4918 pairs) and validate on `datasets/newdata/test` (867 pairs).
- Run 50 epochs with Adam, learning rate `1e-5`, weight decay `1e-5`, cosine scheduling, and 100 optimizer-step linear warmup.
- Keep ResNet18, auxiliary segmentation, `288×384` input, `griding_num=50`, `row_anchor=18`, `num_lanes=1`, and all loss weights unchanged.
- Select the largest safe micro-batch from `8,4,2,1`; use gradient accumulation for effective batch 64.
- Enable CUDA AMP and use two DataLoader workers.
- Include all 4918 samples once per epoch and the 60 `lowlight_camera_full_rgb_` samples two additional times, for 5038 sampled indices per epoch.
- Preserve the old log directory and root ONNX files; all new artifacts go under a new `logs/MMDD_HHMM_lr_1e-05_b_<microbatch>/` directory.
- Report progress after epoch 1, every 5 epochs or 30 minutes, on best-loss updates, on failures, and at completion.
- The workspace is not a Git repository. Replace commit steps with SHA-256/status checkpoints; do not initialize Git.

---

## File Structure

- Create `utils_alian/finetune_utils.py`: deterministic sampler, checkpoint loading, accumulation math, progress/ETA recording, SHA-256 helpers.
- Modify `utils_alian/config.py`: add explicit fine-tuning CLI flags while preserving existing defaults.
- Modify `utils_alian/dataloader_alian.py`: configurable workers and low-light exposure sampler.
- Modify `utils/common.py`: save complete resumable checkpoint state.
- Modify `train_alian.py`: weights-only initialization, AMP, accumulation, per-run summaries, progress JSONL, robust finite-value checks.
- Create `scripts/probe_finetune_memory.py`: real-data forward/loss/backward memory probe for batch candidates.
- Create `scripts/compare_finetuned_models.py`: evaluate old/new models on the 867 test pairs and the 60 low-light training subset.
- Modify `export_onnx.py`: parameterized checkpoint/output paths and ONNX/PyTorch verification.
- Create `tests/test_finetune_utils.py`: sampler, accumulation, ETA, checkpoint-load tests.
- Create `tests/test_finetune_cli.py`: CLI and command-contract tests.
- Create `tests/test_finetune_export.py`: ONNX naming/shape helper tests.
- Generate `logs/<run>/...`: checkpoints, TensorBoard, summaries, progress, comparison, ONNX.

---

### Task 1: Fine-Tuning Utility Contracts

**Files:**
- Create: `utils_alian/finetune_utils.py`
- Test: `tests/test_finetune_utils.py`

**Interfaces:**
- Produces: `LowLightExposureSampler(image_paths, prefix, exposure, seed)` with `set_epoch(epoch)`.
- Produces: `optimizer_updates_per_epoch(loader_batches, accumulation_steps) -> int`.
- Produces: `load_weights_only(model, checkpoint_path) -> dict`.
- Produces: `TrainingProgressRecorder(path, total_epochs)` and `estimate_remaining_seconds(...)`.
- Produces: `sha256_file(path) -> str`.

- [ ] **Step 1: Write failing sampler and accumulation tests**

```python
class SamplerTests(unittest.TestCase):
    def test_all_samples_once_and_low_light_three_times(self):
        paths = ["normal_a.jpg", "lowlight_camera_full_rgb_t000000.jpg",
                 "normal_b.jpg", "lowlight_camera_full_rgb_t000005.jpg"]
        sampler = LowLightExposureSampler(paths, "lowlight_camera_full_rgb_", 3, 123)
        sampler.set_epoch(0)
        values = list(sampler)
        self.assertEqual(len(values), 8)
        self.assertEqual(values.count(0), 1)
        self.assertEqual(values.count(1), 3)
        self.assertEqual(values.count(2), 1)
        self.assertEqual(values.count(3), 3)

    def test_epoch_seed_is_reproducible_but_changes_order(self):
        first = LowLightExposureSampler(make_paths(), "lowlight_", 3, 7)
        second = LowLightExposureSampler(make_paths(), "lowlight_", 3, 7)
        first.set_epoch(4); second.set_epoch(4)
        self.assertEqual(list(first), list(second))
        first.set_epoch(5)
        self.assertNotEqual(list(first), list(second))

class AccumulationTests(unittest.TestCase):
    def test_partial_final_window_counts_as_update(self):
        self.assertEqual(optimizer_updates_per_epoch(79, 16), 5)
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_utils -v`  
Expected: FAIL importing `utils_alian.finetune_utils`.

- [ ] **Step 3: Implement deterministic three-exposure sampling**

```python
class LowLightExposureSampler(torch.utils.data.Sampler[int]):
    def __init__(self, image_paths, prefix, exposure, seed):
        if exposure < 1:
            raise ValueError("exposure must be at least one")
        self.base = list(range(len(image_paths)))
        self.low_light = [i for i, path in enumerate(image_paths)
                          if Path(path).stem.startswith(prefix)]
        if len(self.low_light) != 60:
            raise ValueError(f"expected 60 low-light samples, got {len(self.low_light)}")
        self.exposure = exposure
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        indices = self.base + self.low_light * (self.exposure - 1)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return iter([indices[i] for i in order])

    def __len__(self):
        return len(self.base) + len(self.low_light) * (self.exposure - 1)
```

- [ ] **Step 4: Implement strict weights-only loading and progress helpers**

`load_weights_only` must read `checkpoint["model"]`, strip an optional `module.` prefix, require identical key sets and tensor shapes, load with `strict=True`, and return checkpoint metadata without loading `optimizer`. `TrainingProgressRecorder` writes one flushed JSON object per epoch containing epoch, metrics, elapsed seconds, recent-three-epoch ETA, learning rate, and CUDA peak memory.

- [ ] **Step 5: Run utility tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_utils -v`  
Expected: all Task 1 tests PASS.

- [ ] **Step 6: Record SHA-256 checkpoint**

Run: `Get-FileHash utils_alian\finetune_utils.py,tests\test_finetune_utils.py`  
Expected: two SHA-256 values; no Git commit.

---

### Task 2: CLI, DataLoader, and Training-State Configuration

**Files:**
- Modify: `utils_alian/config.py`
- Modify: `utils_alian/dataloader_alian.py`
- Modify: `utils/common.py`
- Test: `tests/test_finetune_cli.py`

**Interfaces:**
- Adds CLI flags: `--finetune`, `--accumulation_steps`, `--amp`, `--num_workers`, `--low_light_exposure`, `--seed`.
- Extends: `get_train_loader(..., num_workers=2, low_light_exposure=1, seed=20260716)`.
- Extends: `get_val_loader(..., num_workers=2)`.
- Replaces checkpoint writer with `save_training_checkpoint(..., scaler, best_val_loss, global_optimizer_step, options)`.

- [ ] **Step 1: Write failing parser and loader tests**

Test exact defaults and explicit parsing with `unittest.mock.patch("sys.argv", ...)`. Test that a 4918-path dataset with 60 prefixed paths yields sampler length 5038 when exposure is 3.

- [ ] **Step 2: Verify the tests fail on missing arguments**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_cli -v`  
Expected: FAIL because fine-tuning flags and configurable loader arguments do not exist.

- [ ] **Step 3: Add explicit configuration fields**

```python
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--low_light_exposure', type=int, default=1)
parser.add_argument('--seed', type=int, default=20260716)
```

Validate positive batch/accumulation/workers/exposure values and require `batch_size * accumulation_steps == 64` when `finetune` is supplied for this run.

- [ ] **Step 4: Wire the sampler into the training DataLoader**

Use `LowLightExposureSampler` only when `low_light_exposure > 1`; otherwise preserve `RandomSampler`. Set `persistent_workers=num_workers > 0`. Validation remains sequential and unmodified.

- [ ] **Step 5: Save complete resumable state**

Checkpoint dictionary must contain `model`, `optimizer`, `scaler`, `epoch`, `val_loss`, `best_val_loss`, `global_optimizer_step`, and `options`. Write to a temporary sibling path and atomically replace the final `.pth`.

- [ ] **Step 6: Run Task 2 and existing low-light tests**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_cli tests.test_low_light_artifacts tests.test_low_light_dataset_merge -v`  
Expected: all tests PASS.

---

### Task 3: AMP and Gradient-Accumulation Training Loop

**Files:**
- Modify: `train_alian.py`
- Test: `tests/test_finetune_utils.py`

**Interfaces:**
- `train(...) -> dict[str, float]` returns epoch metrics and optimizer update count.
- `val(...) -> dict[str, float]` returns validation loss and metrics.
- Scheduler is constructed with `optimizer_updates_per_epoch` rather than micro-batch count.

- [ ] **Step 1: Add a failing optimizer-step behavior test**

Use a four-batch synthetic model with accumulation 3 and assert exactly two optimizer/scaler/scheduler updates, including the partial final window. Assert logged loss is the unscaled mean rather than the loss divided for accumulation.

- [ ] **Step 2: Run the focused test and verify the old loop fails the contract**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_utils.AccumulatedTrainLoopTests -v`  
Expected: FAIL because the current loop steps on every micro-batch and has no AMP scaler.

- [ ] **Step 3: Implement AMP accumulation**

```python
optimizer.zero_grad(set_to_none=True)
for batch_index, data_label in enumerate(progress_bar):
    with torch.autocast(device_type='cuda', dtype=torch.float16,
                        enabled=options.amp):
        results = inference(net, data_label, options.use_aux)
        raw_loss = calc_loss(...)
        scaled_loss = raw_loss / options.accumulation_steps
    if not torch.isfinite(raw_loss):
        raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_index}")
    scaler.scale(scaled_loss).backward()
    final_batch = batch_index + 1 == len(data_loader)
    update_due = (batch_index + 1) % options.accumulation_steps == 0
    if update_due or final_batch:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(global_optimizer_step)
        global_optimizer_step += 1
```

- [ ] **Step 4: Isolate all run logs inside the work directory**

Replace writes to global `train_log/train_epoch_summary.txt` with `<work_dir>/train_epoch_summary.txt`. Write train and validation metrics to the same run-local summary. Add `training_progress.jsonl` via `TrainingProgressRecorder`. Preserve TensorBoard logging.

- [ ] **Step 5: Load old weights before creating fresh Adam**

Construct the model, call `load_weights_only`, then create a new optimizer. Never pass old optimizer state to the new optimizer. Record source checkpoint hash in `cfg.txt` and `training_result.json`.

- [ ] **Step 6: Run tests and one CPU synthetic smoke test**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest tests.test_finetune_utils tests.test_finetune_cli -v`  
Expected: all tests PASS, exact expected optimizer update counts.

---

### Task 4: Real CUDA Memory Probe

**Files:**
- Create: `scripts/probe_finetune_memory.py`
- Test: `tests/test_finetune_cli.py`

**Interfaces:**
- CLI consumes source, checkpoint, candidates, and model dimensions.
- Produces JSON with candidate status, peak allocated/reserved MiB, selected micro-batch, accumulation steps, and effective batch.

- [ ] **Step 1: Add failing candidate-selection unit tests**

Mock probe results `{8: OOM, 4: pass twice}` and assert selection 4 / accumulation 16. Assert selection fails if no candidate passes twice or 64 is not divisible by the candidate.

- [ ] **Step 2: Implement probe using the real loader and complete loss graph**

For each candidate, construct a non-shuffled one-batch loader, reset CUDA peak stats, load the old weights, create fresh Adam, run forward/loss/backward under AMP twice, and delete all candidate-local objects before testing the next size. Catch only CUDA OOM; propagate data/model errors.

- [ ] **Step 3: Run the real probe**

Run:

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\probe_finetune_memory.py --source datasets\newdata\train --checkpoint logs\0209_1509_lr_1e-04_b_64\best_model.pth --candidates 8 4 2 1 --output datasets\finetune_memory_probe.json
```

Expected: exit 0, selected batch in `{8,4,2,1}`, effective batch 64, and two successful trials for the selected value.

- [ ] **Step 4: Report all actual parameter changes to the user**

Before training, report selected micro-batch, accumulation steps, peak VRAM, effective batch, learning rate, epochs, workers, AMP, three-exposure count, and unchanged model/loss settings.

---

### Task 5: Launch and Monitor the 50-Epoch Fine-Tune

**Files:**
- Generate: `logs/<run>/cfg.txt`
- Generate: `logs/<run>/best_model.pth`
- Generate: `logs/<run>/latest_model.pth`
- Generate: `logs/<run>/train_epoch_summary.txt`
- Generate: `logs/<run>/training_progress.jsonl`
- Generate: `logs/<run>/training_result.json`

**Interfaces:**
- Consumes the batch selected by Task 4.
- Produces a complete resumable training directory and live ETA records.

- [ ] **Step 1: Re-run full tests immediately before training**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v`  
Expected: all tests PASS.

- [ ] **Step 2: Snapshot immutable inputs**

Record SHA-256 for old best checkpoint, dataset baseline/merge receipt, training script, loader, config, constants, and 60 reviewed labels. Confirm counts 4918/4918 and 867/867.

- [ ] **Step 3: Start training in a persistent PTY**

Run with the probe-selected batch `B` and accumulation `64/B`:

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 train_alian.py --source datasets\newdata\train --val_source datasets\newdata\test --log_path logs --epoch 50 --batch_size B --optimizer Adam --learning_rate 1e-5 --weight_decay 1e-5 --scheduler cos --warmup linear --warmup_iters 100 --backbone 18 --griding_num 50 --row_anchor 18 --num_lanes 1 --finetune logs\0209_1509_lr_1e-04_b_64\best_model.pth --accumulation_steps 64/B --amp --num_workers 2 --low_light_exposure 3 --seed 20260716
```

Expected: a new correctly named log directory; printed confirmation that optimizer state was not restored.

- [ ] **Step 4: Establish and report ETA after epoch 1**

Read the first complete JSONL record. Calculate `remaining = moving_average_epoch_seconds × (50 - completed_epochs)` and report metrics, elapsed time, and ETA.

- [ ] **Step 5: Monitor until terminal completion**

Poll the training session without sleeps longer than 60 seconds. Report at least every 5 epochs or 30 minutes and whenever best validation loss improves. If interrupted, resume only from this run's `latest_model.pth`, including optimizer/scaler/global-step state; do not resume from the old baseline checkpoint.

- [ ] **Step 6: Validate training artifacts**

Require 50 train and 50 validation summary records, finite metrics, loadable best/latest checkpoints, and `training_result.json` with total duration, best epoch/loss, final metrics, peak memory, and exact options.

---

### Task 6: New-vs-Old Evaluation

**Files:**
- Create: `scripts/compare_finetuned_models.py`
- Generate: `logs/<run>/comparison_metrics.json`
- Generate: `logs/<run>/comparison_report.md`
- Test: `tests/test_finetune_cli.py`

**Interfaces:**
- Evaluates both checkpoints with identical transforms.
- Produces test-set loss/Top-1/Top-2/Top-3/IoU and low-light loss/accuracy/grid/pixel error.

- [ ] **Step 1: Write failing comparison-report tests**

Given synthetic old/new metric dictionaries, assert the report contains absolute values, signed deltas, percent changes where defined, best epochs, and a warning that the low-light subset participated in training.

- [ ] **Step 2: Implement deterministic evaluation**

Use `simu_transform=None`, sequential sampling, batch equal to the selected safe micro-batch, and `torch.no_grad()` with AMP. Build the low-light subset from formal training indices whose stems start with `lowlight_camera_full_rgb_`; assert exactly 60.

- [ ] **Step 3: Evaluate old and new checkpoints**

Run:

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 scripts\compare_finetuned_models.py --old logs\0209_1509_lr_1e-04_b_64\best_model.pth --new logs\<run>\best_model.pth --test datasets\newdata\test --train datasets\newdata\train --output logs\<run>
```

Expected: JSON and Markdown with both datasets and explicit deltas.

---

### Task 7: Parameterized ONNX Export and Verification

**Files:**
- Modify: `export_onnx.py`
- Create: `tests/test_finetune_export.py`
- Generate: `logs/<run>/resnet_288_384.onnx`
- Generate: `logs/<run>/resnet_288_384_best.onnx`

**Interfaces:**
- CLI: `--checkpoint`, `--output-dir`, `--opset 13`.
- Produces output `(batch,51,18,1)` from input `(batch,3,288,384)`.

- [ ] **Step 1: Write failing parser/naming tests**

Assert output paths are inside the requested run directory and never equal the root legacy ONNX paths. Assert expected input/output shapes.

- [ ] **Step 2: Parameterize export and validate loaded keys**

Load the trained checkpoint's classification/backbone weights into `use_aux=False` model. Permit only known auxiliary-head keys to be unused; fail on missing inference keys. Export opset 13 with dynamic batch.

- [ ] **Step 3: Validate ONNX structure and numerical parity**

Run `onnx.checker.check_model`; compare ONNX Runtime and PyTorch on a fixed-seed input. Require identical output shape and maximum absolute error below `1e-4`. Copy the validated file to both required names inside the run directory and verify SHA-256 equality.

- [ ] **Step 4: Export the final best model**

Run:

```powershell
D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 export_onnx.py --checkpoint logs\<run>\best_model.pth --output-dir logs\<run> --opset 13
```

Expected: two valid run-local ONNX files; root ONNX hashes unchanged.

---

### Task 8: Final Integrity and Delivery Report

**Files:**
- Update: `logs/<run>/training_result.json`
- Update: `logs/<run>/comparison_report.md`

- [ ] **Step 1: Run the full test suite after all generated artifacts exist**

Run: `D:\Anaconda3\envs\hosebot_cv\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v`  
Expected: all tests PASS.

- [ ] **Step 2: Verify checkpoints and ONNX files from disk**

Reload both `.pth` files on CPU, run one PyTorch inference, validate both ONNX files, and compare hashes. Confirm the old log directory and root ONNX hashes match the pre-training snapshot.

- [ ] **Step 3: Verify dataset immutability**

Run the existing merge verifier against `datasets/newdata_before_lowlight.json`; expect train 4918/4918, test 867/867, old hashes unchanged, and no writes under test.

- [ ] **Step 4: Produce the user-facing summary**

Report: every changed parameter and reason, actual micro/effective batch and peak VRAM, total duration, best/final epoch metrics, old/new deltas on 867 test samples, low-light diagnostic deltas and limitation, ONNX parity, and absolute links to every artifact directory/file.

- [ ] **Step 5: Record final source hashes**

Run `Get-FileHash` over modified source/test files and save results in `logs/<run>/source_hashes.json`. The workspace has no Git repository, so no branch/commit/PR action is available.
