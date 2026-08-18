#!/usr/bin/env python3
"""Validate model and dataset inputs for the fixed main-table evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


DATA_FILES = (
    "DocMath/DocMath-Eval/data/complong_testmini-00000-of-00001.parquet",
    "DocMath/DocMath-Eval/data/compshort_testmini-00000-of-00001.parquet",
    "DocMath/DocMath-Eval/data/simplong_testmini-00000-of-00001.parquet",
    "DocMath/DocMath-Eval/data/simpshort_testmini-00000-of-00001.parquet",
    "Frames/test.jsonl",
    "MRCR/mrcr_0_128K.jsonl",
    "CorpusQA/128k_4domains.jsonl",
    "LongBench/Longbench/data/narrativeqa.jsonl",
    "LongBench/Longbench/data/qasper.jsonl",
    "LongBench/Longbench/data/hotpotqa.jsonl",
    "LongBench/Longbench/data/2wikimqa.jsonl",
    "LongBench/Longbench/data/musique.jsonl",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--judge-model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    expected = [
        args.model_path / "config.json",
        args.judge_model_path / "config.json",
        *(args.data_root / relative for relative in DATA_FILES),
    ]
    missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print("Evaluation preflight failed. Missing or empty files:")
        for path in missing:
            print(f"  - {path}")
        return 2

    print(f"Evaluation preflight passed: {len(expected)} required files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
