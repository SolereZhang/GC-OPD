#!/usr/bin/env python3
"""Prepare the exact 32K GoLongRL split used by GC-OPD training.

The paper data pipeline has two stages:

1. append the GRPO output-format instruction and reserve the first 256 rows
   from the ordered GoLongRL shards for validation;
2. render each prompt with the Qwen3 no-thinking chat template, tokenize the
   rendered text exactly as verl does, and retain prompts of at most 32K tokens.

The expected result is 9,527 training rows and 231 validation rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data"
DEFAULT_MODEL_ROOT = REPO_ROOT.parent / "models"

VALIDATION_HOLDOUT_SIZE = 256
MAX_PROMPT_LENGTH = 32_768
EXPECTED_RAW_ROWS = 22_965
EXPECTED_TRAIN_ROWS = 9_527
EXPECTED_VAL_ROWS = 231

GRPO_OUTPUT_INSTRUCTION = """\

Important output format for this GRPO run:
You may reason step by step inside <think>...</think>.
Then put the final answer inside <answer>...</answer>. Preserve the answer format requested by the question, including [Answer] or [答案] when requested, and keep answer items line by line when the question asks for a list.
Do not output anything after </answer>."""

_TOKENIZER = None
_TOKENIZER_PATH = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "GoLongRL" / "data",
        help="Directory containing the downloaded GoLongRL parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "golongrl_32k",
        help="Directory for the filtered train.parquet, val.parquet, and stats.json.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "Qwen3-4B",
        help="Local Qwen3 tokenizer directory used by the student model.",
    )
    parser.add_argument("--num-proc", type=int, default=8, help="Tokenization worker count.")
    parser.add_argument("--batch-size", type=int, default=128, help="Parquet streaming batch size.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument(
        "--no-verify-size",
        action="store_true",
        help="Allow a non-paper dataset revision whose row counts differ from the expected split.",
    )
    return parser.parse_args()


def append_instruction(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append the training output contract to the last user message once."""
    if not prompt:
        return prompt

    target = next((message for message in reversed(prompt) if message.get("role") == "user"), prompt[-1])
    content = target.get("content") or ""
    if "<answer>" not in content:
        target["content"] = content + GRPO_OUTPUT_INSTRUCTION
    return prompt


def discover_shards(input_dir: Path) -> list[Path]:
    """Return the ordered parquet shards from a Hugging Face dataset download."""
    shards = sorted(input_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {input_dir}")
    return shards


def prepare_grpo_split(
    shards: list[Path], output_dir: Path, batch_size: int
) -> tuple[Path, Path, int, int]:
    """Stream the ordered shards into the fixed pre-filter train/val split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    schema = pq.ParquetFile(shards[0]).schema_arrow
    train_writer = pq.ParquetWriter(train_path, schema, compression="zstd")
    val_writer = pq.ParquetWriter(val_path, schema, compression="zstd")

    seen = train_count = val_count = 0
    started = time.time()
    try:
        for shard in shards:
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                train_rows: list[dict[str, Any]] = []
                val_rows: list[dict[str, Any]] = []
                for row in batch.to_pylist():
                    row["prompt"] = append_instruction(row.get("prompt") or [])
                    if seen < VALIDATION_HOLDOUT_SIZE:
                        val_rows.append(row)
                    else:
                        train_rows.append(row)
                    seen += 1

                if train_rows:
                    train_writer.write_table(pa.Table.from_pylist(train_rows, schema=schema))
                    train_count += len(train_rows)
                if val_rows:
                    val_writer.write_table(pa.Table.from_pylist(val_rows, schema=schema))
                    val_count += len(val_rows)

            print(
                f"[prepare] {shard.name}: train={train_count}, val={val_count}, "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
    finally:
        train_writer.close()
        val_writer.close()

    return train_path, val_path, train_count, val_count


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            _TOKENIZER_PATH,
            trust_remote_code=True,
            use_fast=True,
        )
    return _TOKENIZER


def _normalise_messages(messages: Any) -> list[dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return list(messages)


def prompt_length(example: dict[str, Any]) -> dict[str, int]:
    """Mirror the prompt rendering and tokenization in verl's RLHFDataset."""
    tokenizer = _get_tokenizer()
    raw_prompt = tokenizer.apply_chat_template(
        _normalise_messages(example["prompt"]),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    input_ids = tokenizer(raw_prompt, add_special_tokens=False, truncation=False)["input_ids"]
    return {"_prompt_len": len(input_ids)}


def summarise(lengths: list[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {"count": 0, "max_prompt_length": MAX_PROMPT_LENGTH, "kept": 0, "dropped": 0}
    return {
        "count": int(values.size),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "kept": int((values <= MAX_PROMPT_LENGTH).sum()),
        "dropped": int((values > MAX_PROMPT_LENGTH).sum()),
        "min": int(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def filter_split(input_path: Path, output_path: Path, num_proc: int) -> dict[str, Any]:
    dataset = load_dataset("parquet", data_files=str(input_path), split="train")
    with_lengths = dataset.map(
        prompt_length,
        num_proc=num_proc,
        desc=f"Computing prompt lengths for {input_path.name}",
    )
    stats = summarise(with_lengths["_prompt_len"])
    filtered = with_lengths.filter(
        lambda example: example["_prompt_len"] <= MAX_PROMPT_LENGTH,
        num_proc=num_proc,
        desc=f"Keeping prompts <= {MAX_PROMPT_LENGTH} tokens",
    ).remove_columns(["_prompt_len"])
    filtered.to_parquet(str(output_path))
    return stats


def check_output_paths(output_dir: Path, overwrite: bool) -> None:
    output_paths = [output_dir / name for name in ("train.parquet", "val.parquet", "stats.json")]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output files already exist: {names}; pass --overwrite to replace them")
    for path in existing:
        path.unlink()


def verify_counts(raw_rows: int, stats: dict[str, Any]) -> None:
    actual = (raw_rows, stats["train"]["kept"], stats["val"]["kept"])
    expected = (EXPECTED_RAW_ROWS, EXPECTED_TRAIN_ROWS, EXPECTED_VAL_ROWS)
    if actual != expected:
        raise RuntimeError(
            "prepared row counts do not match the paper split: "
            f"got raw/train/val={actual}, expected={expected}. "
            "Check the GoLongRL dataset revision and Qwen3 tokenizer, or pass --no-verify-size "
            "only when intentionally preparing a different dataset revision."
        )


def main() -> None:
    args = parse_args()
    if args.num_proc < 1 or args.batch_size < 1:
        raise ValueError("--num-proc and --batch-size must be positive")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer directory not found: {args.tokenizer_path}")

    shards = discover_shards(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_output_paths(args.output_dir, args.overwrite)

    global _TOKENIZER_PATH
    _TOKENIZER_PATH = str(args.tokenizer_path)

    temporary_parent = args.output_dir.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".golongrl-grpo-", dir=temporary_parent))
    try:
        train_in, val_in, train_before_filter, val_before_filter = prepare_grpo_split(
            shards,
            temporary_dir,
            args.batch_size,
        )
        raw_rows = train_before_filter + val_before_filter
        stats = {
            "dataset": "Kwai-Klear/GoLongRL",
            "preparation": {
                "raw_rows": raw_rows,
                "validation_holdout_rows": VALIDATION_HOLDOUT_SIZE,
                "validation_policy": "first rows in ordered parquet shards",
                "chat_template": "Qwen3, enable_thinking=False",
            },
            "train": filter_split(train_in, args.output_dir / "train.parquet", args.num_proc),
            "val": filter_split(val_in, args.output_dir / "val.parquet", args.num_proc),
        }
        (args.output_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not args.no_verify_size:
            verify_counts(raw_rows, stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[prepare] wrote the 32K split to {args.output_dir}")
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
