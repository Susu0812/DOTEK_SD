import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils_alian.finetune_utils import (
    AccumulationStepper,
    LowLightExposureSampler,
    TrainingProgressRecorder,
    estimate_remaining_seconds,
    load_weights_only,
    optimizer_updates_per_epoch,
)


class SamplerTests(unittest.TestCase):
    def test_all_samples_once_and_low_light_three_times(self):
        paths = [
            "normal_a.jpg",
            "lowlight_camera_full_rgb_t000000.jpg",
            "normal_b.jpg",
            "lowlight_camera_full_rgb_t000005.jpg",
        ]
        sampler = LowLightExposureSampler(
            paths,
            prefix="lowlight_camera_full_rgb_",
            exposure=3,
            seed=123,
            expected_low_light_count=2,
        )
        sampler.set_epoch(0)
        values = list(sampler)

        self.assertEqual(len(values), 8)
        self.assertEqual(values.count(0), 1)
        self.assertEqual(values.count(1), 3)
        self.assertEqual(values.count(2), 1)
        self.assertEqual(values.count(3), 3)

    def test_epoch_seed_is_reproducible_but_changes_order(self):
        paths = [
            "normal_a.jpg",
            "lowlight_camera_full_rgb_t000000.jpg",
            "normal_b.jpg",
        ]
        first = LowLightExposureSampler(
            paths, "lowlight_camera_full_rgb_", 3, 7,
            expected_low_light_count=1,
        )
        second = LowLightExposureSampler(
            paths, "lowlight_camera_full_rgb_", 3, 7,
            expected_low_light_count=1,
        )
        first.set_epoch(4)
        second.set_epoch(4)
        self.assertEqual(list(first), list(second))

        first.set_epoch(5)
        self.assertNotEqual(list(first), list(second))

    def test_wrong_low_light_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 60"):
            LowLightExposureSampler(
                ["normal.jpg"],
                "lowlight_camera_full_rgb_",
                3,
                1,
            )


class AccumulationTests(unittest.TestCase):
    def test_partial_final_window_counts_as_update(self):
        self.assertEqual(optimizer_updates_per_epoch(79, 16), 5)

    def test_non_positive_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            optimizer_updates_per_epoch(0, 16)
        with self.assertRaises(ValueError):
            optimizer_updates_per_epoch(79, 0)

    def test_four_micro_batches_with_accumulation_three_update_twice(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        class CountingScheduler:
            def __init__(self):
                self.steps = []

            def step(self, value):
                self.steps.append(value)

        scheduler = CountingScheduler()
        stepper = AccumulationStepper(
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            accumulation_steps=3,
        )
        updates = []
        for batch_index in range(4):
            loss = model(torch.ones(1, 1)).sum()
            updates.append(
                stepper.backward_and_maybe_step(
                    loss,
                    batch_index=batch_index,
                    total_batches=4,
                )
            )

        self.assertEqual(updates, [False, False, True, True])
        self.assertEqual(stepper.global_optimizer_step, 2)
        self.assertEqual(scheduler.steps, [0, 1])


class WeightLoadingTests(unittest.TestCase):
    def test_loads_model_weights_without_optimizer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = torch.nn.Linear(3, 2)
            target = torch.nn.Linear(3, 2)
            with torch.no_grad():
                source.weight.fill_(2.5)
                source.bias.fill_(-0.5)
            path = Path(directory) / "best_model.pth"
            torch.save(
                {
                    "model": source.state_dict(),
                    "optimizer": {"must_not_load": True},
                    "epoch": 196,
                    "val_loss": 0.1908,
                },
                path,
            )

            metadata = load_weights_only(target, path)

            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertTrue(torch.equal(target.bias, source.bias))
            self.assertEqual(metadata["epoch"], 196)
            self.assertNotIn("optimizer", metadata)

    def test_shape_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            torch.save({"model": torch.nn.Linear(4, 2).state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_weights_only(torch.nn.Linear(3, 2), path)


class ProgressTests(unittest.TestCase):
    def test_eta_uses_recent_three_complete_epochs(self):
        eta = estimate_remaining_seconds([10.0, 20.0, 30.0, 40.0], 4, 10)
        self.assertEqual(eta, 180.0)

    def test_recorder_writes_one_flushed_json_object_per_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_progress.jsonl"
            recorder = TrainingProgressRecorder(path, total_epochs=50)
            recorder.record(
                epoch=1,
                train_metrics={"loss": 0.3},
                val_metrics={"loss": 0.2},
                epoch_seconds=12.0,
                elapsed_seconds=12.0,
                learning_rate=1e-5,
                peak_memory_mib=1500.0,
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["epoch"], 1)
            self.assertEqual(records[0]["remaining_epochs"], 49)
            self.assertEqual(records[0]["eta_seconds"], 588.0)


if __name__ == "__main__":
    unittest.main()
