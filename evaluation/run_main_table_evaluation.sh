#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "This fixed entrypoint accepts no positional arguments." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

required_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "$name is required." >&2
    exit 2
  fi
}

for name in MODEL_PATH JUDGE_MODEL_PATH DATA_ROOT OUTPUT_DIR; do
  required_env "$name"
done

export PUBLIC_EVAL_ROOT="$SCRIPT_DIR"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export MODEL_NAME=main_table_candidate
export RUN_ID=main_table_five_task
export OUT_ROOT="$OUTPUT_DIR/predictions"
export JUDGE_OUT_ROOT="$OUTPUT_DIR/judge"

# Runtime parallelism does not change the evaluation protocol.
export VLLM_TP="${MODEL_TP:-8}"
export JUDGE_TP="${JUDGE_TP:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
export JUDGE_VISIBLE_DEVICES="${JUDGE_VISIBLE_DEVICES:-$($PYTHON_BIN - "$JUDGE_TP" <<'PY'
import sys

print(",".join(str(index) for index in range(int(sys.argv[1]))))
PY
)}"

# The paper protocol is intentionally immutable in this entrypoint.
export TARGET_BENCHMARKS=docmath,frames,mrcr,corpusqa,lbv1qa
export MAX_INPUT_TOKENS=120000
export MAX_OUTPUT_TOKENS=8192
export MAX_MODEL_LEN=131072
export YARN_FACTOR=4
export QWEN3_THINKING_MODE=nothink
export JUDGE_MAX_MODEL_LEN=32768
export JUDGE_MAX_TOKENS=2048
export JUDGE_QWEN3_THINKING_MODE=auto
export SMOKE_LIMIT=0
export RESUME_MODE=auto

print_protocol() {
  printf '%s\n' \
    "protocol=main_table_five_task_v1" \
    "benchmarks=$TARGET_BENCHMARKS" \
    "model_path=$MODEL_PATH" \
    "judge_model_path=$JUDGE_MODEL_PATH" \
    "data_root=$DATA_ROOT" \
    "output_dir=$OUTPUT_DIR" \
    "max_input_tokens=$MAX_INPUT_TOKENS" \
    "max_output_tokens=$MAX_OUTPUT_TOKENS" \
    "max_model_len=$MAX_MODEL_LEN" \
    "yarn_factor=$YARN_FACTOR" \
    "thinking_mode=$QWEN3_THINKING_MODE" \
    "temperature=0.7" \
    "top_p=0.95" \
    "judge_max_model_len=$JUDGE_MAX_MODEL_LEN" \
    "judge_max_tokens=$JUDGE_MAX_TOKENS" \
    "model_tp=$VLLM_TP" \
    "judge_tp=$JUDGE_TP"
}

print_protocol
if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

"$PYTHON_BIN" "$SCRIPT_DIR/check_data_layout.py" \
  --model-path "$MODEL_PATH" \
  --judge-model-path "$JUDGE_MODEL_PATH" \
  --data-root "$DATA_ROOT"

mkdir -p "$OUT_ROOT" "$JUDGE_OUT_ROOT"
cd "$SCRIPT_DIR/harness"
bash scripts/main_table/predict_then_judge.sh
