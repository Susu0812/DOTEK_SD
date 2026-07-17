import subprocess
import sys
import unittest
from pathlib import Path

from scripts.build_low_light_training_dataset import build_parser


class CliTests(unittest.TestCase):
    def test_script_path_can_import_project_package(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_low_light_training_dataset.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="replace",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare", completed.stdout)

    def test_all_non_training_phases_are_available(self):
        parser = build_parser()
        cases = [
            ["prepare", "--video", "v.mp4", "--output", "out", "--checkpoint", "m.pth"],
            ["finalize", "--output", "out", "--review", "review.json"],
            ["validate", "--output", "out"],
            ["snapshot", "--dataset", "newdata", "--output", "base.json"],
            ["merge", "--output", "out", "--dataset", "newdata"],
            ["verify-merge", "--output", "out", "--dataset", "newdata", "--baseline", "base.json"],
        ]

        parsed = [parser.parse_args(arguments).command for arguments in cases]

        self.assertEqual(
            parsed,
            ["prepare", "finalize", "validate", "snapshot", "merge", "verify-merge"],
        )


if __name__ == "__main__":
    unittest.main()
