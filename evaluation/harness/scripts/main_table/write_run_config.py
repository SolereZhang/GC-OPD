#!/usr/bin/env python3
"""Write a reproducible, non-secret run config snapshot for main-table jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


COMMON_ENV_KEYS = [
    "RUN_ID",
    "RESUME_ID",
    "RESUME_MODE",
    "ATTEMPT_ID",
    "SMOKE_LIMIT",
    "EVAL_PRESET",
    "LITE_SEED",
]

PREDICT_ENV_KEYS = COMMON_ENV_KEYS + [
    "MODEL_NAME",
    "TARGET_BENCHMARKS",
    "EVAL_STAGE",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MAX_MODEL_LEN",
    "TARGET_CONTEXT_LENGTH",
    "YARN_TARGET_CONTEXT_LENGTH",
    "YARN_FACTOR",
    "YARN_MODE",
    "YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS",
    "QWEN3_THINKING_MODE",
    "VLLM_CHAT_TEMPLATE_KWARGS",
    "VLLM_TP",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "PORT",
    "HF_ENDPOINT",
]

JUDGE_ENV_KEYS = COMMON_ENV_KEYS + [
    "PRED_MODEL_NAME",
    "MODEL_NAME",
    "JUDGE_TARGETS",
    "JUDGE_MODEL_NAME",
    "JUDGE_PORT",
    "JUDGE_TP",
    "JUDGE_MAX_MODEL_LEN",
    "JUDGE_MAX_TOKENS",
    "JUDGE_GPU_MEMORY_UTILIZATION",
    "JUDGE_QWEN3_THINKING_MODE",
    "JUDGE_VLLM_CHAT_TEMPLATE_KWARGS",
]

IDENTITY_ENV_KEYS = {
    "predict": [
        "MODEL_PATH",
        "DATA_ROOT",
        "MODEL_NAME",
        "TARGET_BENCHMARKS",
        "MAX_INPUT_TOKENS",
        "MAX_OUTPUT_TOKENS",
        "MAX_MODEL_LEN",
        "YARN_FACTOR",
        "QWEN3_THINKING_MODE",
    ],
    "judge": [
        "PRED_OUT",
        "PRED_MODEL_NAME",
        "JUDGE_MODEL_PATH",
        "JUDGE_MODEL_NAME",
        "JUDGE_TARGETS",
        "JUDGE_MAX_MODEL_LEN",
        "JUDGE_MAX_TOKENS",
        "JUDGE_QWEN3_THINKING_MODE",
    ],
}

PATH_ENV_KEYS = {"MODEL_PATH", "DATA_ROOT", "PRED_OUT", "JUDGE_MODEL_PATH"}

PACKAGE_NAMES = [
    "vllm",
    "evalscope",
    "transformers",
    "torch",
    "openai",
    "datasets",
    "modelscope",
]


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_env(keys: list[str]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if key in os.environ}


def write_env_file(path: Path, env: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(env):
            value = env[key].replace("\\", "\\\\").replace('"', '\\"')
            handle.write(f'{key}="{value}"\n')


def run_identity(kind: str) -> dict[str, object]:
    values = {}
    for key in IDENTITY_ENV_KEYS[kind]:
        if key not in os.environ:
            continue
        value = os.environ[key]
        if key in PATH_ENV_KEYS:
            value = str(Path(value).expanduser().resolve())
        values[key] = value
    return {"schema_version": 1, "kind": kind, "inputs": values}


def ensure_run_identity(out: Path, identity: dict[str, object]) -> None:
    path = out / "run_identity.json"
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot read existing run identity: {path}: {error}") from error
        if previous != identity:
            raise SystemExit(
                f"Refusing to reuse {out} for different evaluation inputs. "
                "Choose a new OUTPUT_DIR or restore the original inputs."
            )
        return

    if any(out.glob("*.exit")):
        raise SystemExit(
            f"Cannot safely resume {out}: completion markers exist without run_identity.json. "
            "Choose a new OUTPUT_DIR."
        )
    path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["predict", "judge"], required=True)
    parser.add_argument("--phase", default="initial")
    args = parser.parse_args()

    out_text = os.environ.get("OUT")
    if not out_text:
        raise SystemExit("OUT is required")

    out = Path(out_text)
    attempt_id = os.environ.get("ATTEMPT_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    attempts = out / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)

    keys = PREDICT_ENV_KEYS if args.kind == "predict" else JUDGE_ENV_KEYS
    env = collect_env(keys)
    ensure_run_identity(out, run_identity(args.kind))
    config = {
        "schema_version": 1,
        "kind": f"main_table_{args.kind}",
        "phase": args.phase,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "packages": package_versions(),
        },
        "env": env,
    }

    config_text = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out / "run_config.json").write_text(config_text, encoding="utf-8")
    (attempts / f"{attempt_id}.run_config.json").write_text(config_text, encoding="utf-8")
    write_env_file(out / "run_config.env", env)
    write_env_file(attempts / f"{attempt_id}.run_config.env", env)
    print(f"[config] wrote {out / 'run_config.json'} and attempt snapshot for {attempt_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
