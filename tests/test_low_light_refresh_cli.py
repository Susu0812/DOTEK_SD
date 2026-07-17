import subprocess
import sys
import unittest
from pathlib import Path

from scripts.refresh_low_light_candidates import build_parser


class RefreshCliTests(unittest.TestCase):
    def test_all_commands_parse(self):
        parser = build_parser()
        commands = [
            parser.parse_args([
                "quarantine", "--bundle", "b", "--dataset", "d", "--output", "q"
            ]).command,
            parser.parse_args([
                "extract", "--video", "v", "--output", "o"
            ]).command,
            parser.parse_args([
                "verify", "--dataset", "d", "--quarantine", "q", "--candidates", "c"
            ]).command,
        ]
        self.assertEqual(commands, ["quarantine", "extract", "verify"])

    def test_script_help_runs_from_project_root(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "refresh_low_light_candidates.py"), "--help"],
            cwd=root, capture_output=True, text=True,
            encoding="gbk", errors="replace", check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("quarantine", completed.stdout)


if __name__ == "__main__":
    unittest.main()

