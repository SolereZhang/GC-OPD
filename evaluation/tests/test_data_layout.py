from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "check_data_layout.py"
SPEC = importlib.util.spec_from_file_location("check_data_layout", MODULE_PATH)
assert SPEC and SPEC.loader
check_data_layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_data_layout)


class DataLayoutTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_path = self.root / "candidate"
        self.judge_model_path = self.root / "judge"
        self.data_root = self.root / "data"
        for path in (
            self.model_path / "config.json",
            self.judge_model_path / "config.json",
            *(self.data_root / relative for relative in check_data_layout.DATA_FILES),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_preflight(self):
        return subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--model-path",
                str(self.model_path),
                "--judge-model-path",
                str(self.judge_model_path),
                "--data-root",
                str(self.data_root),
            ],
            text=True,
            capture_output=True,
        )

    def test_complete_layout_passes(self):
        completed = self.run_preflight()
        self.assertEqual(completed.returncode, 0)
        self.assertIn("14 required files found", completed.stdout)

    def test_missing_dataset_file_fails(self):
        (self.data_root / check_data_layout.DATA_FILES[0]).unlink()
        completed = self.run_preflight()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(check_data_layout.DATA_FILES[0], completed.stdout)


if __name__ == "__main__":
    unittest.main()
