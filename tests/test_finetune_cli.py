import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils.common import save_training_checkpoint
from utils_alian.config import get_args, validate_finetune_options
from utils_alian.dataloader_alian import build_train_sampler
from utils_alian.finetune_utils import LowLightExposureSampler, select_safe_batch


class ParserTests(unittest.TestCase):
    def test_finetune_arguments_are_explicitly_parsed(self):
        argv = [
            "train_alian.py",
            "--batch_size", "4",
            "--finetune", "old_best.pth",
            "--accumulation_steps", "16",
            "--amp",
            "--num_workers", "2",
            "--low_light_exposure", "3",
            "--seed", "20260716",
        ]
        with patch("sys.argv", argv):
            options = get_args()

        self.assertEqual(options.batch_size, 4)
        self.assertEqual(options.finetune, "old_best.pth")
        self.assertEqual(options.accumulation_steps, 16)
        self.assertTrue(options.amp)
        self.assertEqual(options.num_workers, 2)
        self.assertEqual(options.low_light_exposure, 3)
        self.assertEqual(options.seed, 20260716)

    def test_effective_batch_must_equal_64_for_finetune(self):
        argv = [
            "train_alian.py",
            "--batch_size", "4",
            "--finetune", "old_best.pth",
            "--accumulation_steps", "8",
        ]
        with patch("sys.argv", argv):
            options = get_args()
        with self.assertRaisesRegex(ValueError, "effective batch size 64"):
            validate_finetune_options(options)


class FakeDataset:
    def __init__(self):
        self.img_paths = [f"normal_{index:04d}.jpg" for index in range(4858)]
        self.img_paths.extend(
            f"lowlight_camera_full_rgb_t{index * 5:06d}.jpg"
            for index in range(60)
        )

    def __len__(self):
        return len(self.img_paths)


class LoaderTests(unittest.TestCase):
    def test_three_exposure_sampler_has_5038_indices(self):
        sampler = build_train_sampler(
            FakeDataset(),
            distributed=False,
            low_light_exposure=3,
            seed=20260716,
        )

        self.assertIsInstance(sampler, LowLightExposureSampler)
        self.assertEqual(len(sampler), 5038)


class MemoryProbeSelectionTests(unittest.TestCase):
    def test_selects_first_candidate_with_two_successful_trials(self):
        selected = select_safe_batch(
            [
                {"batch_size": 8, "trials": [False]},
                {"batch_size": 4, "trials": [True, True]},
                {"batch_size": 2, "trials": [True, True]},
            ],
            effective_batch_size=64,
        )
        self.assertEqual(selected["batch_size"], 4)
        self.assertEqual(selected["accumulation_steps"], 16)

    def test_rejects_candidates_without_two_passes(self):
        with self.assertRaisesRegex(RuntimeError, "no safe micro-batch"):
            select_safe_batch(
                [{"batch_size": 4, "trials": [True, False]}],
                effective_batch_size=64,
            )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_contains_complete_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            output = Path(directory) / "latest_model.pth"

            save_training_checkpoint(
                output,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=7,
                val_loss=0.2,
                best_val_loss=0.19,
                global_optimizer_step=640,
                options={"batch_size": 4, "accumulation_steps": 16},
            )

            checkpoint = torch.load(output, map_location="cpu")
            self.assertEqual(
                set(checkpoint),
                {
                    "model", "optimizer", "scaler", "epoch", "val_loss",
                    "best_val_loss", "global_optimizer_step", "options",
                },
            )
            self.assertEqual(checkpoint["epoch"], 7)
            self.assertEqual(checkpoint["global_optimizer_step"], 640)
            self.assertFalse(output.with_suffix(".pth.tmp").exists())


if __name__ == "__main__":
    unittest.main()
