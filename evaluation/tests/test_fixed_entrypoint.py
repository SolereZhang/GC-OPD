from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "evaluation"
ENTRYPOINT = ROOT / "evaluation" / "run_main_table_evaluation.sh"
BENCHMARK_ROOT = EVALUATION_ROOT / "harness" / "benchmarks"
MAIN_TABLE_SCRIPTS = EVALUATION_ROOT / "harness" / "scripts" / "main_table"


class FixedEntrypointTest(unittest.TestCase):
    def test_dry_run_prints_only_fixed_protocol(self):
        env = os.environ.copy()
        env.update(
            {
                "MODEL_PATH": "/tmp/candidate",
                "JUDGE_MODEL_PATH": "/tmp/judge",
                "DATA_ROOT": "/tmp/data",
                "OUTPUT_DIR": "/tmp/output",
                "DRY_RUN": "1",
            }
        )
        completed = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn(
            "benchmarks=docmath,frames,mrcr,corpusqa,lbv1qa", completed.stdout
        )
        self.assertIn("max_input_tokens=120000", completed.stdout)
        self.assertIn("thinking_mode=nothink", completed.stdout)

    def test_positional_override_is_rejected(self):
        completed = subprocess.run(
            ["bash", str(ENTRYPOINT), "--benchmark", "mrcr"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)

    def test_all_custom_evaluator_paths_exist(self):
        expected = ("mrcr_eval.py", "corpusqa_eval.py", "lbv1qa_eval.py")
        scripts = "\n".join(
            (MAIN_TABLE_SCRIPTS / name).read_text(encoding="utf-8")
            for name in ("predict.sh", "judge.sh")
        )
        self.assertNotIn("$BENCH_ROOT/evalscope/", scripts)
        for filename in expected:
            self.assertTrue((BENCHMARK_ROOT / filename).is_file(), filename)
            self.assertIn(f"$BENCH_ROOT/{filename}", scripts)

    def test_evaluation_tree_has_no_suite_or_provenance_markers(self):
        prohibited = (
            "qwen" "long",
            "longbench" "-v2",
            "longbench" "_v2",
            "lb" "v2",
            "upstream " "commit",
            "sha" "-256",
        )
        for path in EVALUATION_ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".py", ".sh", ".txt"}:
                continue
            content = path.read_text(encoding="utf-8").lower()
            for marker in prohibited:
                self.assertNotIn(marker, content, str(path))


if __name__ == "__main__":
    unittest.main()
