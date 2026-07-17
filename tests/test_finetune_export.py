import tempfile
import unittest
from pathlib import Path

import torch

from scripts.onnx_export_utils import prepare_inference_state, resolve_output_paths


class ExportPathTests(unittest.TestCase):
    def test_both_onnx_outputs_stay_inside_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            standard, explicit = resolve_output_paths(root)

            self.assertEqual(standard, root / "resnet_288_384.onnx")
            self.assertEqual(explicit, root / "resnet_288_384_best.onnx")
            self.assertEqual(standard.parent, root)
            self.assertEqual(explicit.parent, root)

    def test_training_auxiliary_keys_are_ignored_but_core_keys_are_required(self):
        target = {
            "model.weight": torch.zeros(2, 2),
            "cls.bias": torch.zeros(2),
        }
        checkpoint = {
            "module.model.weight": torch.ones(2, 2),
            "module.cls.bias": torch.ones(2),
            "module.aux_header2.0.weight": torch.ones(1),
            "module.aux_combine.0.weight": torch.ones(1),
        }

        prepared = prepare_inference_state(checkpoint, target)
        self.assertEqual(set(prepared), set(target))
        self.assertTrue(torch.equal(prepared["model.weight"], torch.ones(2, 2)))

    def test_missing_or_mismatched_core_keys_are_rejected(self):
        target = {"model.weight": torch.zeros(2, 2)}
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_inference_state({}, target)
        with self.assertRaisesRegex(ValueError, "shape"):
            prepare_inference_state({"model.weight": torch.zeros(1, 2)}, target)


if __name__ == "__main__":
    unittest.main()
