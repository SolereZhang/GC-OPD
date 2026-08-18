from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "aggregate_main_table.py"
SPEC = importlib.util.spec_from_file_location("aggregate_main_table", MODULE_PATH)
assert SPEC and SPEC.loader
aggregate_main_table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_main_table)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class AggregateMainTableTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pred_out = self.root / "predictions"
        self.judge_out = self.root / "judge"
        self.model_name = "candidate"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_complete_fixture(self):
        write_json(
            self.pred_out / "docmath" / "reports" / self.model_name / "docmath.json",
            {"name": "docmath", "model_name": self.model_name, "score": 0.4},
        )
        write_json(
            self.pred_out / "frames" / "reports" / self.model_name / "frames.json",
            {"name": "frames", "model_name": self.model_name, "score": 0.5},
        )
        write_json(
            self.pred_out / "runs" / f"{self.model_name}_mrcr_128k_summary.json",
            {"overall": 60},
        )
        write_json(
            self.judge_out / "judge_summary.json",
            {
                "corpusqa": {"num": 10, "correct": 7, "score": 70},
                "lbv1qa": {
                    "subsets": {
                        "narrativeqa": 80,
                        "qasper": 80,
                        "hotpotqa": 80,
                        "2wikimqa": 80,
                        "musique": 80,
                    },
                    "overall": 80,
                },
            },
        )

    def test_fixed_five_task_average(self):
        self.write_complete_fixture()
        summary = aggregate_main_table.aggregate(
            self.pred_out, self.judge_out, self.model_name
        )

        self.assertEqual(
            tuple(summary["protocol"]["benchmarks"]),
            aggregate_main_table.BENCHMARKS,
        )
        self.assertEqual(summary["protocol"]["name"], "main_table_five_task_v1")
        self.assertEqual(summary["missing"], [])
        self.assertAlmostEqual(summary["main_table_average"], 0.6)
        self.assertAlmostEqual(summary["main_table_average_percent"], 60.0)
        self.assertNotIn("pred_out", summary)
        self.assertNotIn("judge_out", summary)
        for item in summary["benchmarks"].values():
            if item["source"] is not None:
                self.assertFalse(Path(item["source"]).is_absolute())

    def test_missing_score_never_produces_average(self):
        self.write_complete_fixture()
        (self.judge_out / "judge_summary.json").unlink()

        summary = aggregate_main_table.aggregate(
            self.pred_out, self.judge_out, self.model_name
        )

        self.assertEqual(summary["missing"], ["corpusqa", "lbv1qa"])
        self.assertIsNone(summary["main_table_average"])
        self.assertIsNone(summary["main_table_average_percent"])

    def test_score_normalization_uses_explicit_scale(self):
        self.assertEqual(aggregate_main_table.normalize_score(1, "fraction"), 1)
        self.assertEqual(aggregate_main_table.normalize_score(1, "percent"), 0.01)
        self.assertEqual(aggregate_main_table.normalize_score(25, "percent"), 0.25)
        with self.assertRaises(ValueError):
            aggregate_main_table.normalize_score(25, "fraction")
        with self.assertRaises(ValueError):
            aggregate_main_table.normalize_score(101, "percent")

    def test_other_models_report_is_not_used_as_fallback(self):
        self.write_complete_fixture()
        (self.pred_out / "docmath" / "reports" / self.model_name / "docmath.json").unlink()
        write_json(
            self.pred_out / "docmath" / "reports" / "stale-model" / "docmath.json",
            {"score": 0.99},
        )

        summary = aggregate_main_table.aggregate(
            self.pred_out, self.judge_out, self.model_name
        )

        self.assertIn("docmath", summary["missing"])
        self.assertIsNone(summary["benchmarks"]["docmath"]["score"])
        self.assertIsNone(summary["main_table_average"])

    def test_partial_lbv1qa_subsets_do_not_count_as_complete(self):
        self.write_complete_fixture()
        summary_path = self.judge_out / "judge_summary.json"
        judge_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del judge_summary["lbv1qa"]["subsets"]["musique"]
        write_json(summary_path, judge_summary)

        summary = aggregate_main_table.aggregate(
            self.pred_out, self.judge_out, self.model_name
        )

        self.assertIn("lbv1qa", summary["missing"])
        self.assertIsNone(summary["benchmarks"]["lbv1qa"]["score"])

    def test_inconsistent_lbv1qa_overall_is_rejected(self):
        self.write_complete_fixture()
        summary_path = self.judge_out / "judge_summary.json"
        judge_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        judge_summary["lbv1qa"]["overall"] = 79
        write_json(summary_path, judge_summary)

        with self.assertRaises(ValueError):
            aggregate_main_table.aggregate(
                self.pred_out, self.judge_out, self.model_name
            )

    def test_paper_gc_opd_scores_reproduce_reported_average(self):
        self.write_complete_fixture()
        write_json(
            self.pred_out / "docmath" / "reports" / self.model_name / "docmath.json",
            {"score": 0.5550},
        )
        write_json(
            self.pred_out / "frames" / "reports" / self.model_name / "frames.json",
            {"score": 0.3459},
        )
        write_json(
            self.pred_out / "runs" / f"{self.model_name}_mrcr_128k_summary.json",
            {"overall": 31.10},
        )
        write_json(
            self.judge_out / "judge_summary.json",
            {
                "corpusqa": {"score": 43.7690},
                "lbv1qa": {
                    "subsets": {
                        "narrativeqa": 58.30,
                        "qasper": 58.30,
                        "hotpotqa": 58.30,
                        "2wikimqa": 58.30,
                        "musique": 58.30,
                    },
                    "overall": 58.30,
                },
            },
        )

        summary = aggregate_main_table.aggregate(
            self.pred_out, self.judge_out, self.model_name
        )

        self.assertEqual(summary["missing"], [])
        self.assertAlmostEqual(summary["main_table_average"], 0.446518)

    def test_cli_writes_json_and_csv(self):
        self.write_complete_fixture()
        output = self.root / "scores.json"
        csv_output = self.root / "scores.csv"

        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--pred-out",
                str(self.pred_out),
                "--judge-out",
                str(self.judge_out),
                "--model-name",
                self.model_name,
                "--out",
                str(output),
                "--csv",
                str(csv_output),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("**60.00**", completed.stdout)
        self.assertTrue(output.is_file())
        self.assertTrue(csv_output.is_file())
        self.assertEqual(json.loads(output.read_text())["missing"], [])


if __name__ == "__main__":
    unittest.main()
