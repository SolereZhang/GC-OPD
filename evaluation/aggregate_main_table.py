#!/usr/bin/env python3
"""Aggregate the fixed five-task protocol reported in the main table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARKS = ("docmath", "frames", "mrcr", "corpusqa", "lbv1qa")
LBV1QA_SUBSETS = (
    "narrativeqa",
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "musique",
)


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def numeric_field(obj: Any, key: str) -> float | None:
    if not isinstance(obj, dict) or key not in obj or obj[key] is None:
        return None
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return value


def normalize_score(value: float | None, scale: str) -> float | None:
    if value is None:
        return None
    if scale == "fraction":
        if value < 0 or value > 1:
            raise ValueError(f"fraction score outside [0, 1]: {value}")
        return value
    if scale == "percent":
        if value < 0 or value > 100:
            raise ValueError(f"percent score outside [0, 100]: {value}")
        return value / 100.0
    raise ValueError(f"unsupported score scale: {scale}")


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def result(
    source: Path | None,
    source_root: Path,
    raw_score: float | None,
    score_scale: str,
    count: int | None,
):
    relative_source = None
    if source is not None:
        try:
            relative_source = str(source.relative_to(source_root))
        except ValueError:
            relative_source = source.name
    return {
        "source": relative_source,
        "raw_score": raw_score,
        "score": normalize_score(raw_score, score_scale),
        "num_predictions": count,
    }


def collect_evalscope(
    pred_out: Path,
    model_name: str,
    benchmark: str,
    report_names: list[str],
) -> dict[str, Any]:
    root = pred_out / benchmark
    report = first_existing(
        [root / "reports" / model_name / name for name in report_names]
    )
    prediction_count = 0
    prediction_root = root / "predictions" / model_name
    if prediction_root.is_dir():
        prediction_count = sum(
            len(load_jsonl(path)) for path in prediction_root.rglob("*.jsonl")
        )
    return result(
        report,
        pred_out,
        numeric_field(load_json(report), "score") if report else None,
        "fraction",
        prediction_count or None,
    )


def collect_mrcr(pred_out: Path, model_name: str) -> dict[str, Any]:
    report = first_existing(
        [
            pred_out / "runs" / f"{model_name}_mrcr_128k_summary.json",
            pred_out / "evals" / f"{model_name}_mrcr_128k_summary.json",
        ]
    )
    predictions = load_jsonl(
        pred_out / "runs" / f"{model_name}_mrcr_128k.jsonl"
    )
    return result(
        report,
        pred_out,
        numeric_field(load_json(report), "overall") if report else None,
        "percent",
        len(predictions) or None,
    )


def collect_judged(judge_out: Path, benchmark: str) -> dict[str, Any]:
    summary_path = judge_out / "judge_summary.json"
    summary = load_json(summary_path)
    item = summary.get(benchmark) if isinstance(summary, dict) else None
    count = item.get("num") if isinstance(item, dict) else None
    raw_score = numeric_field(item, "score") if benchmark == "corpusqa" else None
    if benchmark == "lbv1qa" and isinstance(item, dict):
        subsets = item.get("subsets")
        if isinstance(subsets, dict) and all(name in subsets for name in LBV1QA_SUBSETS):
            subset_scores = [numeric_field(subsets, name) for name in LBV1QA_SUBSETS]
            if all(score is not None for score in subset_scores):
                raw_score = sum(subset_scores) / len(LBV1QA_SUBSETS)
                reported = numeric_field(item, "overall")
                if reported is not None and not math.isclose(
                    reported, raw_score, rel_tol=0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"lbv1qa overall {reported} does not match subset mean {raw_score}"
                    )
    return result(
        summary_path if item is not None else None,
        judge_out,
        raw_score,
        "percent",
        count,
    )


def aggregate(pred_out: Path, judge_out: Path, model_name: str) -> dict[str, Any]:
    benchmarks = {
        "docmath": collect_evalscope(
            pred_out, model_name, "docmath", ["docmath.json", "report.json"]
        ),
        "frames": collect_evalscope(
            pred_out, model_name, "frames", ["frames.json", "report.json"]
        ),
        "mrcr": collect_mrcr(pred_out, model_name),
        "corpusqa": collect_judged(judge_out, "corpusqa"),
        "lbv1qa": collect_judged(judge_out, "lbv1qa"),
    }
    missing = [name for name in BENCHMARKS if benchmarks[name]["score"] is None]
    scores = [benchmarks[name]["score"] for name in BENCHMARKS if name not in missing]
    average = sum(scores) / len(BENCHMARKS) if not missing else None
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "main_table_five_task_v1",
            "benchmarks": list(BENCHMARKS),
            "average": "unweighted_arithmetic_mean",
            "score_scale": "normalized_0_to_1",
        },
        "model_name": model_name,
        "benchmarks": benchmarks,
        "missing": missing,
        "main_table_average": average,
        "main_table_average_percent": average * 100 if average is not None else None,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark", "raw_score", "score", "score_percent", "num_predictions", "source"),
        )
        writer.writeheader()
        for name in BENCHMARKS:
            item = summary["benchmarks"][name]
            score = item["score"]
            writer.writerow(
                {
                    "benchmark": name,
                    "raw_score": item["raw_score"],
                    "score": score,
                    "score_percent": score * 100 if score is not None else None,
                    "num_predictions": item["num_predictions"],
                    "source": item["source"],
                }
            )


def print_summary(summary: dict[str, Any]) -> None:
    print("| Benchmark | Score (%) | Predictions |")
    print("|---|---:|---:|")
    for name in BENCHMARKS:
        item = summary["benchmarks"][name]
        score = item["score"]
        score_text = f"{score * 100:.2f}" if score is not None else "missing"
        count = item["num_predictions"] or "-"
        print(f"| {name} | {score_text} | {count} |")
    average = summary["main_table_average_percent"]
    average_text = f"{average:.2f}" if average is not None else "incomplete"
    print(f"| **Avg.** | **{average_text}** | - |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-out", type=Path, required=True)
    parser.add_argument("--judge-out", type=Path, required=True)
    parser.add_argument("--model-name", default="main_table_candidate")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    output = args.out or args.judge_out / "main_table_scores.json"
    csv_path = args.csv or args.judge_out / "main_table_scores.csv"
    summary = aggregate(args.pred_out, args.judge_out, args.model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, summary)
    print_summary(summary)
    print(f"Wrote {output} and {csv_path}")
    if summary["missing"] and not args.allow_partial:
        print("Missing required benchmarks: " + ", ".join(summary["missing"]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
