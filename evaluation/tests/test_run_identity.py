from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "evaluation" / "harness" / "scripts" / "main_table" / "write_run_config.py"


class RunIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.out = self.root / "output"
        self.env = os.environ.copy()
        self.env.update(
            {
                "OUT": str(self.out),
                "ATTEMPT_ID": "test-attempt",
                "MODEL_PATH": str(self.root / "model-a"),
                "DATA_ROOT": str(self.root / "data"),
                "MODEL_NAME": "main_table_candidate",
                "TARGET_BENCHMARKS": "docmath,frames,mrcr,corpusqa,lbv1qa",
                "MAX_INPUT_TOKENS": "120000",
                "MAX_OUTPUT_TOKENS": "8192",
                "MAX_MODEL_LEN": "131072",
                "YARN_FACTOR": "4",
                "QWEN3_THINKING_MODE": "nothink",
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_config(self, env=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--kind", "predict", "--phase", "initial"],
            cwd=ROOT,
            env=env or self.env,
            text=True,
            capture_output=True,
        )

    def test_same_inputs_can_resume(self):
        self.assertEqual(self.run_config().returncode, 0)
        self.assertEqual(self.run_config().returncode, 0)
        self.assertTrue((self.out / "run_identity.json").is_file())

    def test_different_model_cannot_reuse_output(self):
        self.assertEqual(self.run_config().returncode, 0)
        changed = self.env.copy()
        changed["MODEL_PATH"] = str(self.root / "model-b")

        completed = self.run_config(changed)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("different evaluation inputs", completed.stderr)

    def test_completion_marker_without_identity_is_rejected(self):
        self.out.mkdir(parents=True)
        (self.out / "docmath_smoke.exit").write_text("0\n", encoding="utf-8")

        completed = self.run_config()

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("without run_identity.json", completed.stderr)


if __name__ == "__main__":
    unittest.main()
