#!/usr/bin/env bash
set -euo pipefail

: "${PUBLIC_EVAL_ROOT:?PUBLIC_EVAL_ROOT must point to the evaluation directory}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
python() {
  "$PYTHON_BIN" "$@"
}
export -f python

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -c 'from evalscope import run_task; import vllm'

pick_free_port() {
  python - "$1" "$2" <<'PYPORT'
import random
import socket
import sys

lo = int(sys.argv[1])
hi = int(sys.argv[2])
for port in random.sample(range(lo, hi + 1), min(4096, hi - lo + 1)):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    break
else:
    raise SystemExit(f"No free port found in [{lo}, {hi}]")
PYPORT
}

export MODEL_NAME="${MODEL_NAME:-main_table_candidate}"
export EVAL_PRESET=full
export BENCH_ROOT="$PUBLIC_EVAL_ROOT/harness/benchmarks"
export PORT="${PORT:-$(pick_free_port 18000 23999)}"
export MAX_INPUT_TOKENS=120000
export MAX_OUTPUT_TOKENS=8192
export YARN_TARGET_CONTEXT_LENGTH=131072
export MAX_MODEL_LEN=131072
export TARGET_CONTEXT_LENGTH="$MAX_MODEL_LEN"
export VLLM_TP="${VLLM_TP:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
export YARN_MODE=auto
export YARN_FACTOR=4
export QWEN3_THINKING_MODE=nothink
export TARGET_BENCHMARKS=docmath,frames,mrcr,corpusqa,lbv1qa
export EVAL_STAGE=predict
export RESUME_MODE=auto
export RUN_ID="${RUN_ID:-main_table_five_task}"
case "$EVAL_STAGE" in
  full|predict|judge) ;;
  *)
    echo "[job] invalid EVAL_STAGE=$EVAL_STAGE; expected full, predict, or judge"
    exit 2
    ;;
esac

if [ -z "${OUT:-}" ]; then
  case "$RESUME_MODE" in
    auto|force|true|1|yes)
      if [ -z "$RUN_ID" ]; then
        RUN_ID="$(python - "$MODEL_NAME" "$TARGET_BENCHMARKS" "$EVAL_STAGE" "$MAX_INPUT_TOKENS" "$MAX_OUTPUT_TOKENS" "$YARN_FACTOR" <<'PYID'
import re
import sys

model, benches, stage, max_in, max_out, yarn = sys.argv[1:7]
raw = f"{model}-{stage}-{benches}-in{max_in}_out{max_out}_yarn{yarn}"
safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", raw).strip("._-")
print(safe[:180] or "main_table_resume_run")
PYID
)"
      fi
      OUT="$OUT_ROOT/$RUN_ID"
      ;;
    *)
      OUT="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)-${MODEL_NAME}"
      ;;
  esac
fi
export OUT RUN_ID
if [ -z "${ATTEMPT_ID:-}" ]; then
  ATTEMPT_HOST="$(hostname 2>/dev/null | tr -c 'A-Za-z0-9._=-' '_' | cut -c1-40)"
  ATTEMPT_ID="$(date +%Y%m%d_%H%M%S)-${ATTEMPT_HOST:-host}-pid$$-r${RANDOM:-0}"
fi
export ATTEMPT_ID

want_bench() {
  case ",${TARGET_BENCHMARKS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

step_name_for_bench() {
  case "$1" in
    mrcr) echo "mrcr_128k_smoke" ;;
    docmath) echo "docmath_smoke" ;;
    frames) echo "frames_smoke" ;;
    corpusqa) echo "corpusqa_128k_smoke" ;;
    lbv1qa) echo "lbv1qa_smoke" ;;
    *) return 1 ;;
  esac
}

resume_enabled() {
  case "$RESUME_MODE" in
    disable|disabled|false|0|no) return 1 ;;
    *) return 0 ;;
  esac
}

step_exit_success() {
  local name="$1"
  resume_enabled || return 1
  [ -f "$OUT/${name}.exit" ] || return 1
  [ "$(cat "$OUT/${name}.exit" 2>/dev/null)" = "0" ]
}

requested_steps_complete() {
  local bench step any=0
  for bench in mrcr docmath frames corpusqa lbv1qa; do
    if want_bench "$bench"; then
      any=1
      step="$(step_name_for_bench "$bench")"
      step_exit_success "$step" || return 1
    fi
  done
  [ "$any" = "1" ]
}

print_exit_summary() {
  echo "===== SUMMARY exits ====="
  for f in "$OUT"/*.exit; do
    [ -f "$f" ] || continue
    echo "$(basename "$f" .exit): $(cat "$f")"
  done
}

mkdir -p "$OUT" "$OUT/data" "$OUT/runs" "$OUT/evals" "$OUT/model"
mkdir -p "$OUT/attempts"
export ATTEMPT_LOG="$OUT/attempts/${ATTEMPT_ID}.log"
export VLLM_LOG="$OUT/attempts/${ATTEMPT_ID}.vllm.log"
export EVALSCOPE_RUNNER="$OUT/attempts/run_evalscope_task_${ATTEMPT_ID}.py"
exec > >(tee -a "$OUT/job.log" "$ATTEMPT_LOG") 2>&1

echo "[job] out=$OUT"
echo "[job] out_root=$OUT_ROOT run_id=${RUN_ID:-} resume_mode=$RESUME_MODE attempt_id=$ATTEMPT_ID"
echo "[job] attempt_log=$ATTEMPT_LOG"
echo "[job] vllm_log=$VLLM_LOG"
echo "[job] model=$MODEL_PATH"
echo "[job] bench=$BENCH_ROOT"
echo "[job] eval_preset=$EVAL_PRESET"
echo "[job] data=$DATA_ROOT"
echo "[job] python=$(which python)"
echo "[job] python_version=$(python --version 2>&1)"
echo "[job] targets=$TARGET_BENCHMARKS"
echo "[job] eval_stage=$EVAL_STAGE"
echo "[job] port=$PORT max_model_len=$MAX_MODEL_LEN max_input_tokens=$MAX_INPUT_TOKENS max_output_tokens=$MAX_OUTPUT_TOKENS yarn_target=$YARN_TARGET_CONTEXT_LENGTH tp=$VLLM_TP qwen3_thinking_mode=$QWEN3_THINKING_MODE"
python scripts/main_table/write_run_config.py --kind predict --phase initial

if requested_steps_complete; then
  echo "[resume] all requested steps already have successful exits; exiting without starting vLLM"
  print_exit_summary
  echo "[job] done out=$OUT"
  exit 0
fi

test -d "$MODEL_PATH"
test -d "$BENCH_ROOT"
test -d "$DATA_ROOT"

python - <<'PYYARN'
import json
import math
import os
from pathlib import Path

src = Path(os.environ["MODEL_PATH"])
out = Path(os.environ["OUT"])
server_max = int(os.environ["MAX_MODEL_LEN"])
yarn_target = int(os.environ["YARN_TARGET_CONTEXT_LENGTH"])
mode = os.environ.get("YARN_MODE", "auto").lower()
cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
max_pos = int(cfg.get("max_position_embeddings") or 0)
needs_yarn = mode == "force" or (mode == "auto" and max_pos and max_pos < yarn_target)
effective = src


def link_model_tree(source: Path, target: Path, skip: set[str] | None = None) -> None:
    skip = skip or set()
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in skip:
            continue
        link = target / item.name
        if not link.exists():
            link.symlink_to(item)


def tokenizer_config_needs_sanitize(model_dir: Path) -> bool:
    path = model_dir / "tokenizer_config.json"
    if not path.exists():
        return False
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(cfg.get("extra_special_tokens"), list)


def sanitize_tokenizer_config(model_dir: Path) -> bool:
    path = model_dir / "tokenizer_config.json"
    if not path.exists():
        return False
    cfg = json.loads(path.read_text(encoding="utf-8"))
    extra = cfg.get("extra_special_tokens")
    if not isinstance(extra, list):
        return False
    additional = cfg.get("additional_special_tokens")
    merged = list(additional) if isinstance(additional, list) else []
    for token in extra:
        if token not in merged:
            merged.append(token)
    if merged:
        cfg["additional_special_tokens"] = merged
    cfg.pop("extra_special_tokens", None)
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[model] sanitized tokenizer_config extra_special_tokens for {model_dir}")
    return True


if needs_yarn:
    dst = out / "model" / (src.name + f"_yarn_{server_max}")
    link_model_tree(src, dst, skip={"config.json"})
    original = int(os.environ.get("YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS") or max_pos or 32768)
    factor = float(os.environ.get("YARN_FACTOR") or math.ceil((yarn_target / original) * 100) / 100)
    cfg["max_position_embeddings"] = max(server_max, max_pos)
    cfg["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original,
    }
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sanitize_tokenizer_config(dst)
    effective = dst
    print(f"[yarn] enabled original={original} factor={factor} yarn_target={yarn_target} max_model_len={server_max} patched_model={effective}")
else:
    if tokenizer_config_needs_sanitize(src):
        dst = out / "model" / (src.name + "_tokenizer_sanitized")
        link_model_tree(src, dst)
        sanitize_tokenizer_config(dst)
        effective = dst
        print(f"[model] tokenizer_config overlay={effective}")
    print(f"[yarn] disabled max_position_embeddings={max_pos} yarn_target={yarn_target} max_model_len={server_max}")

(out / "effective_model_path.txt").write_text(str(effective), encoding="utf-8")
PYYARN

EFFECTIVE_MODEL_PATH="$(cat "$OUT/effective_model_path.txt")"
echo "[job] effective_model=$EFFECTIVE_MODEL_PATH"

wait_chat_server() {
  for i in $(seq 1 240); do
    if curl -s -o /dev/null -w "%{http_code}" \
      -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' | grep -q 200; then
      echo "[vllm] ready on port $PORT"
      return 0
    fi
    sleep 5
  done
  echo "[vllm] startup timeout on port $PORT"
  return 1
}

kill_process_tree() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_process_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    if [ "${VLLM_OWN_SESSION:-0}" = "1" ]; then
      kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    fi
    kill_process_tree "$VLLM_PID"
    sleep 2
    if [ "${VLLM_OWN_SESSION:-0}" = "1" ]; then
      kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
    fi
    for child in $(pgrep -P "$VLLM_PID" 2>/dev/null || true); do
      kill -9 "$child" 2>/dev/null || true
    done
    kill -9 "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}
trap cleanup EXIT

VLLM_CHAT_TEMPLATE_ARGS=()
case "$(printf '%s' "$QWEN3_THINKING_MODE" | tr '[:upper:]' '[:lower:]' | tr '-' '_')" in
  auto|"")
    QWEN3_THINKING_MODE_NORM=auto
    ;;
  think|thinking|enable|enabled|true|1|yes)
    QWEN3_THINKING_MODE_NORM=think
    VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
    ;;
  nothink|no_think|non_thinking|disable|disabled|false|0|no)
    QWEN3_THINKING_MODE_NORM=nothink
    VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
    ;;
  *)
    echo "[vllm] invalid QWEN3_THINKING_MODE=$QWEN3_THINKING_MODE; expected auto, think, or nothink"
    exit 2
    ;;
esac

if [ -n "${VLLM_CHAT_TEMPLATE_KWARGS:-}" ]; then
  if python -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- "--default-chat-template-kwargs"; then
    VLLM_CHAT_TEMPLATE_ARGS=(--default-chat-template-kwargs "$VLLM_CHAT_TEMPLATE_KWARGS")
    export VLLM_CHAT_TEMPLATE_KWARGS
    echo "[vllm] default_chat_template_kwargs=$VLLM_CHAT_TEMPLATE_KWARGS"
  elif [ "$QWEN3_THINKING_MODE_NORM" = "nothink" ]; then
    EFFECTIVE_MODEL_PATH="$(python scripts/main_table/patch_qwen3_chat_template.py \
      --model-path "$EFFECTIVE_MODEL_PATH" \
      --out "$OUT" \
      --mode "$QWEN3_THINKING_MODE_NORM")"
    echo "[vllm] default-chat-template-kwargs unsupported; using patched no-think model overlay: $EFFECTIVE_MODEL_PATH"
  else
    echo "[vllm] default-chat-template-kwargs unsupported; QWEN3_THINKING_MODE=think is equivalent to Qwen3 default for this template"
  fi
fi
export EFFECTIVE_MODEL_PATH
python scripts/main_table/write_run_config.py --kind predict --phase resolved

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(python - "$VLLM_TP" <<'PYCUDALIST'
import sys

tp = int(sys.argv[1])
print(",".join(str(i) for i in range(tp)))
PYCUDALIST
)"
  export CUDA_VISIBLE_DEVICES
fi
echo "[vllm] cuda_visible_devices=$CUDA_VISIBLE_DEVICES tensor_parallel_size=$VLLM_TP"

VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$EFFECTIVE_MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$VLLM_TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --port "$PORT" \
  "${VLLM_CHAT_TEMPLATE_ARGS[@]}"
)
if command -v setsid >/dev/null 2>&1; then
  setsid "${VLLM_CMD[@]}" > "$VLLM_LOG" 2>&1 &
  VLLM_OWN_SESSION=1
else
  "${VLLM_CMD[@]}" > "$VLLM_LOG" 2>&1 &
  VLLM_OWN_SESSION=0
fi
VLLM_PID=$!

echo "[vllm] pid=$VLLM_PID waiting..."
if ! wait_chat_server; then
  tail -260 "$VLLM_LOG" || true
  exit 1
fi

line_count() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return 0
  fi
  wc -l < "$path" | tr -d ' '
}

smoke_limit_enabled() {
  [ "${SMOKE_LIMIT:-0}" -gt 0 ] 2>/dev/null
}

step_display_name() {
  local name="$1"
  local suffix=""
  if smoke_limit_enabled; then
    suffix="_smoke${SMOKE_LIMIT}"
  fi

  case "$name" in
    mrcr_128k_smoke) echo "mrcr_128k${suffix}" ;;
    docmath_smoke) echo "docmath${suffix}" ;;
    frames_smoke) echo "frames${suffix}" ;;
    corpusqa_128k_smoke) echo "corpusqa_128k${suffix}" ;;
    lbv1qa_smoke) echo "lbv1qa${suffix}" ;;
    *) echo "$name" ;;
  esac
}

run_step() {
  name="$1"; shift
  local display
  display="$(step_display_name "$name")"
  if step_exit_success "$name"; then
    echo "===== STEP ${display} SKIP completed exit=0 (state=${name}) ====="
    return 0
  fi
  echo "===== STEP ${display} START (state=${name}) ====="
  set +e
  ( "$@" ) > "$OUT/${name}.log" 2>&1
  code=$?
  set -e
  echo "===== STEP ${display} EXIT ${code} (state=${name}) ====="
  tail -180 "$OUT/${name}.log" || true
  echo "$code" > "$OUT/${name}.exit"
}

require_jsonl_count() {
  local name="$1"
  local expected_path="$2"
  local actual_path="$3"
  local expected actual
  expected="$(line_count "$expected_path")"
  actual="$(line_count "$actual_path")"
  echo "[check] ${name}: expected_lines=${expected} actual_lines=${actual} file=${actual_path}" | tee -a "$OUT/job.log"
  if [ "$expected" = "0" ] || [ "$actual" != "$expected" ]; then
    echo "[check] ${name}: incomplete prediction output" | tee -a "$OUT/job.log"
    echo 3 > "$OUT/${name}.exit"
    return 1
  fi
  return 0
}

jsonl_tree_line_count() {
  local dir="$1"
  python - "$dir" <<'PYCOUNT'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
total = 0
if root.is_dir():
    for path in root.rglob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            total += sum(1 for _ in handle)
print(total)
PYCOUNT
}

parquet_tree_row_count() {
  local dir="$1"
  python - "$dir" <<'PYPARQUETCOUNT'
import pathlib
import sys

import pyarrow.parquet as pq

root = pathlib.Path(sys.argv[1])
print(sum(pq.ParquetFile(path).metadata.num_rows for path in root.rglob("*.parquet")))
PYPARQUETCOUNT
}

print_step_progress() {
  local name="$1"
  local total="$2"
  local existing="$3"
  local input_path="${4:-}"
  local output_path="${5:-}"
  local detail="${6:-}"
  local display remaining
  display="$(step_display_name "$name")"

  if [[ "$total" =~ ^[0-9]+$ ]] && [[ "$existing" =~ ^[0-9]+$ ]]; then
    remaining=$((total - existing))
    if [ "$remaining" -lt 0 ]; then
      remaining=0
    fi
  else
    remaining="unknown"
  fi

  echo "[progress] ${display}: total=${total} existing=${existing} remaining=${remaining} resume_mode=${RESUME_MODE} state=${name}${detail:+ ${detail}}"
  [ -n "$input_path" ] && echo "[progress] ${display}: input=${input_path}"
  [ -n "$output_path" ] && echo "[progress] ${display}: output=${output_path}"
}

require_jsonl_tree_count() {
  local name="$1"
  local expected_dir="$2"
  local actual_dir="$3"
  local expected actual
  expected="$(jsonl_tree_line_count "$expected_dir")"
  actual="$(jsonl_tree_line_count "$actual_dir")"
  echo "[check] ${name}: expected_lines=${expected} actual_lines=${actual} dir=${actual_dir}" | tee -a "$OUT/job.log"
  if [ "$expected" = "0" ] || [ "$actual" != "$expected" ]; then
    echo "[check] ${name}: incomplete prediction output" | tee -a "$OUT/job.log"
    echo 3 > "$OUT/${name}.exit"
    return 1
  fi
  return 0
}

require_count_value() {
  local name="$1"
  local expected="$2"
  local actual="$3"
  echo "[check] ${name}: expected_records=${expected} actual_records=${actual}" | tee -a "$OUT/job.log"
  if [ "$expected" = "0" ] || [ "$actual" != "$expected" ]; then
    echo "[check] ${name}: incomplete prediction output" | tee -a "$OUT/job.log"
    echo 3 > "$OUT/${name}.exit"
    return 1
  fi
  return 0
}

want_bench() {
  case ",${TARGET_BENCHMARKS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

write_evalscope_runner() {
  cat > "$EVALSCOPE_RUNNER" <<'PYEVAL'
import argparse
import json

from evalscope import run_task
from evalscope.config import TaskConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--work_dir", required=True)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--dataset_args", default="{}")
parser.add_argument("--judge_api_key", default="")
parser.add_argument("--judge_model_id", default="qwen3-30b-a3b-instruct-2507")
parser.add_argument("--judge_api_url", default="http://127.0.0.1:1/v1")
parser.add_argument("--judge_strategy", default="auto")
parser.add_argument("--max_tokens", type=int, default=51200)
parser.add_argument("--eval_batch_size", type=int, default=8)
args = parser.parse_args()

dataset_args = json.loads(args.dataset_args)


def build_context_truncator(tokenizer_path, max_input_tokens):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def chat_token_len(text):
        messages = [{"role": "user", "content": text}]
        return len(tok.apply_chat_template(messages, add_generation_prompt=True))

    def truncate_text(text, render_prompt):
        prompt = render_prompt(text)
        if chat_token_len(prompt) <= max_input_tokens:
            return text

        text_ids = tok.encode(text, add_special_tokens=False)
        lo, hi = 0, len(text_ids)
        best = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            left = mid // 2
            right = mid - left
            candidate = tok.decode(text_ids[:left] + text_ids[-right:], skip_special_tokens=True)
            if chat_token_len(render_prompt(candidate)) <= max_input_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    return truncate_text


def patch_docmath_truncation(dataset_args):
    tokenizer_path = dataset_args.get("tokenizer_path")
    max_input_tokens = int(dataset_args.get("max_input_tokens") or 0)
    if args.dataset != "docmath" or not tokenizer_path or max_input_tokens <= 0:
        return

    from evalscope.api.dataset import Sample
    from evalscope.api.messages import ChatMessageUser
    from evalscope.benchmarks.docmath.docmath_adapter import DocMathAdapter, TEMPLATE_0SHOT

    truncate_text = build_context_truncator(tokenizer_path, max_input_tokens)

    def record_to_sample(self, record):
        ground_truth = record["ground_truth"]
        context = "\n".join(record["paragraphs"])
        question = record["question"]
        context = truncate_text(context, lambda c: TEMPLATE_0SHOT.format(context=c, question=question))
        message = self.prompt_template.format(context=context, question=question)
        return Sample(
            input=[ChatMessageUser(content=message)],
            target=str(ground_truth),
            metadata={
                "question_id": record.get("question_id", ""),
                "answer_type": type(ground_truth).__name__,
                "question": question,
            },
        )

    DocMathAdapter.record_to_sample = record_to_sample
    print(f"[docmath] patched tokenizer truncation max_input_tokens={max_input_tokens} tokenizer={tokenizer_path}")


def patch_frames_truncation(dataset_args):
    tokenizer_path = dataset_args.get("tokenizer_path")
    max_input_tokens = int(dataset_args.get("max_input_tokens") or 0)
    if args.dataset != "frames" or not tokenizer_path or max_input_tokens <= 0:
        return

    from evalscope.benchmarks.frames.frames_adapter import FramesAdapter, TEMPLATE_0SHOT

    truncate_text = build_context_truncator(tokenizer_path, max_input_tokens)

    def format_prompt_template(self, sample):
        context = sample.metadata["context"]
        question = sample.input
        context = truncate_text(context, lambda c: TEMPLATE_0SHOT.format(context=c, question=question))
        return self.prompt_template.format(context=context, question=question)

    FramesAdapter.format_prompt_template = format_prompt_template
    print(f"[frames] patched tokenizer truncation max_input_tokens={max_input_tokens} tokenizer={tokenizer_path}")


patch_docmath_truncation(dataset_args)
patch_frames_truncation(dataset_args)

judge_model_args = None
if args.judge_api_key:
    judge_model_args = {
        "model_id": args.judge_model_id,
        "api_url": args.judge_api_url,
        "api_key": args.judge_api_key,
    }

task_cfg = TaskConfig(
    model=args.model_name,
    api_url=f"http://127.0.0.1:{args.port}/v1",
    api_key="EMPTY",
    datasets=[args.dataset],
    dataset_args={args.dataset: dataset_args},
    generation_config={
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    },
    judge_model_args=judge_model_args,
    judge_strategy=args.judge_strategy,
    repeats=1,
    eval_batch_size=args.eval_batch_size,
    use_cache=args.work_dir,
    work_dir=args.work_dir,
    limit=args.limit,
)
run_task(task_cfg=task_cfg)
PYEVAL
}
write_evalscope_runner

LIMIT_ARGS=()
if [ "$SMOKE_LIMIT" -gt 0 ] 2>/dev/null; then
  LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
fi

JUDGE_API_KEY="${JUDGE_API_KEY:-}"

evalscope_judge_args() {
  if [ "$EVAL_STAGE" = "predict" ]; then
    printf '%s\n' --judge_strategy
    printf '%s\n' rule
    return 0
  fi
  if [ -z "$JUDGE_API_KEY" ]; then
    return 1
  fi
  printf '%s\n' --judge_api_key
  printf '%s\n' "$JUDGE_API_KEY"
  printf '%s\n' --judge_model_id
  printf '%s\n' "${JUDGE_MODEL_ID:-qwen3-30b-a3b-instruct-2507}"
  printf '%s\n' --judge_api_url
  printf '%s\n' "${JUDGE_API_URL:-http://127.0.0.1:1/v1}"
  printf '%s\n' --judge_strategy
  printf '%s\n' auto
}

if want_bench mrcr; then
  MRCR_SRC="${MRCR_DATA:-}"
  if [ -z "$MRCR_SRC" ]; then
    if [ -f "$DATA_ROOT/MRCR/mrcr_0_128K.jsonl" ]; then
      MRCR_SRC="$DATA_ROOT/MRCR/mrcr_0_128K.jsonl"
    else
      MRCR_SRC="$DATA_ROOT/MRCR/mrcr_128k_smoke.jsonl"
    fi
  fi
  MRCR_RUN_DATA="$OUT/data/mrcr_128k_smoke.jsonl"
  if [ -f "$MRCR_SRC" ]; then
    if [ "$SMOKE_LIMIT" -gt 0 ] 2>/dev/null; then
      head -n "$SMOKE_LIMIT" "$MRCR_SRC" > "$MRCR_RUN_DATA"
    else
      MRCR_RUN_DATA="$MRCR_SRC"
    fi
    MRCR_OUT="$OUT/runs/${MODEL_NAME}_mrcr_128k.jsonl"
    print_step_progress mrcr_128k_smoke \
      "$(line_count "$MRCR_RUN_DATA")" \
      "$(line_count "$MRCR_OUT")" \
      "$MRCR_RUN_DATA" \
      "$MRCR_OUT"
    run_step mrcr_128k_smoke python "$BENCH_ROOT/mrcr_eval.py" \
      --data_file "$MRCR_RUN_DATA" \
      --model "$MODEL_NAME" \
      --model_path "$EFFECTIVE_MODEL_PATH" \
      --api_url "http://127.0.0.1:${PORT}/v1" \
      --api_key EMPTY \
      --concurrency 4 \
      --max_tokens "$MAX_OUTPUT_TOKENS" \
      --timeout 3600 \
      --infer_output "$MRCR_OUT" \
      --max_input_tokens "$MAX_INPUT_TOKENS"
    require_jsonl_count mrcr_128k_smoke "$MRCR_RUN_DATA" "$MRCR_OUT" || true
  else
    echo "[mrcr] missing local data: $MRCR_SRC"
    echo 2 > "$OUT/mrcr_128k_smoke.exit"
  fi
fi

if want_bench docmath; then
  DOCMATH_JUDGE_ARGS=()
  if ! mapfile -t DOCMATH_JUDGE_ARGS < <(evalscope_judge_args); then
    echo "[docmath] skip: missing judge key"
    echo 99 > "$OUT/docmath_smoke.exit"
  else
    print_step_progress docmath_smoke \
      "$(jsonl_tree_line_count "$DATA_ROOT/DocMath/DocMath-Eval")" \
      "$(jsonl_tree_line_count "$OUT/docmath/predictions/$MODEL_NAME")" \
      "$DATA_ROOT/DocMath/DocMath-Eval" \
      "$OUT/docmath/predictions/$MODEL_NAME" \
      "source_jsonl_lines_only=1"
    run_step docmath_smoke python "$EVALSCOPE_RUNNER" \
      --dataset docmath \
      --model_name "$MODEL_NAME" \
      --port "$PORT" \
      --work_dir "$OUT/docmath" \
      "${DOCMATH_JUDGE_ARGS[@]}" \
      --dataset_args '{"dataset_id":"'"$DATA_ROOT"'/DocMath/DocMath-Eval","subset_list":["complong_testmini","compshort_testmini","simplong_testmini","simpshort_testmini"],"filters":{"remove_until":"</think>"},"tokenizer_path":"'"$EFFECTIVE_MODEL_PATH"'","max_input_tokens":'"${DOCMATH_MAX_INPUT_TOKENS:-$MAX_INPUT_TOKENS}"'}' \
      --max_tokens "$MAX_OUTPUT_TOKENS" \
      --eval_batch_size 32 \
      "${LIMIT_ARGS[@]}"
    if ! smoke_limit_enabled; then
      require_count_value docmath_smoke \
        "$(parquet_tree_row_count "$DATA_ROOT/DocMath/DocMath-Eval/data")" \
        "$(jsonl_tree_line_count "$OUT/docmath/predictions/$MODEL_NAME")" || true
    fi
  fi
fi

if want_bench frames; then
  FRAMES_JUDGE_ARGS=()
  if ! mapfile -t FRAMES_JUDGE_ARGS < <(evalscope_judge_args); then
    echo "[frames] skip: missing judge key"
    echo 99 > "$OUT/frames_smoke.exit"
  else
    if [ ! -f "$DATA_ROOT/Frames/test.jsonl" ]; then
      echo "[frames] missing local data: $DATA_ROOT/Frames/test.jsonl"
      echo 2 > "$OUT/frames_smoke.exit"
    else
      FRAMES_ARGS='{"dataset_id":"'"$DATA_ROOT"'/Frames","filters":{"remove_until":"</think>"},"max_length":'"$MAX_INPUT_TOKENS"',"truncation_strategy":"middle","tokenizer_path":"'"$EFFECTIVE_MODEL_PATH"'","max_input_tokens":'"$MAX_INPUT_TOKENS"'}'
      print_step_progress frames_smoke \
        "$(line_count "$DATA_ROOT/Frames/test.jsonl")" \
        "$(jsonl_tree_line_count "$OUT/frames/predictions/$MODEL_NAME")" \
        "$DATA_ROOT/Frames/test.jsonl" \
        "$OUT/frames/predictions/$MODEL_NAME"
      run_step frames_smoke python "$EVALSCOPE_RUNNER" \
        --dataset frames \
        --model_name "$MODEL_NAME" \
        --port "$PORT" \
        --work_dir "$OUT/frames" \
        "${FRAMES_JUDGE_ARGS[@]}" \
        --dataset_args "$FRAMES_ARGS" \
        --max_tokens "$MAX_OUTPUT_TOKENS" \
        --eval_batch_size 8 \
        "${LIMIT_ARGS[@]}"
      if ! smoke_limit_enabled; then
        require_count_value frames_smoke \
          "$(line_count "$DATA_ROOT/Frames/test.jsonl")" \
          "$(jsonl_tree_line_count "$OUT/frames/predictions/$MODEL_NAME")" || true
      fi
    fi
  fi
fi

if want_bench corpusqa; then
  if [ "$EVAL_STAGE" != "predict" ] && [ -z "$JUDGE_API_KEY" ]; then
    echo "[corpusqa] skip: missing judge key"
    echo 99 > "$OUT/corpusqa_128k_smoke.exit"
  else
    CORPUSQA_SRC="${CORPUSQA_DATA:-$DATA_ROOT/CorpusQA/128k_4domains.jsonl}"
    CORPUSQA_DATA="$OUT/data/corpusqa_128k_smoke.jsonl"
    if [ -f "$CORPUSQA_SRC" ]; then
      python - "$CORPUSQA_SRC" "$CORPUSQA_DATA" "$MAX_INPUT_TOKENS" "$EFFECTIVE_MODEL_PATH" "$SMOKE_LIMIT" <<'PYCORPUSQA'
import json
import sys

from transformers import AutoTokenizer

src, dst, max_tokens_text, model_path, limit_text = sys.argv[1:6]
max_tokens = int(max_tokens_text)
limit = int(limit_text)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def n_tokens_messages(messages):
    return len(tok.apply_chat_template(messages, add_generation_prompt=True))

def middle_truncate_messages(messages):
    if max_tokens <= 0 or not messages or n_tokens_messages(messages) <= max_tokens:
        return messages

    query = messages[-1]
    query_tokens = len(tok.encode(query.get("content", ""), add_special_tokens=False))
    budget = max_tokens - query_tokens - 1024
    if budget <= 0:
        return [query]

    prefix_text = "\n".join(m.get("content", "") for m in messages[:-1])
    prefix_tokens = tok.encode(prefix_text, add_special_tokens=False)
    if len(prefix_tokens) > budget:
        half = max(0, budget // 2)
        prefix_tokens = prefix_tokens[:half] + prefix_tokens[-half:]
        prefix_text = tok.decode(prefix_tokens, skip_special_tokens=True)
    return [{"role": "user", "content": prefix_text}, query]

written = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if limit > 0 and written >= limit:
            break
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get("prompt", [])
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(prompt, list):
            item["prompt"] = middle_truncate_messages(prompt)
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1
print(f"[corpusqa] wrote {written} records to {dst} with max_input_tokens={max_tokens}")
PYCORPUSQA
      CORPUSQA_OUT="$OUT/runs/${MODEL_NAME}_corpusqa_128k.jsonl"
      print_step_progress corpusqa_128k_smoke \
        "$(line_count "$CORPUSQA_DATA")" \
        "$(line_count "$CORPUSQA_OUT")" \
        "$CORPUSQA_DATA" \
        "$CORPUSQA_OUT"
      CORPUSQA_CMD=(
        python "$BENCH_ROOT/corpusqa_eval.py" \
        --data_file "$CORPUSQA_DATA" \
        --model "$MODEL_NAME" \
        --api_url "http://127.0.0.1:${PORT}/v1" \
        --api_key EMPTY \
        --concurrency 4 \
        --max_tokens "$MAX_OUTPUT_TOKENS" \
        --timeout 3600 \
        --infer_output "$CORPUSQA_OUT" \
        --eval_output "$OUT/evals/${MODEL_NAME}_corpusqa_128k_eval.jsonl" \
        --judge_api_key "${JUDGE_API_KEY:-EMPTY}" \
        --judge_base_url "${JUDGE_BASE_URL:-${JUDGE_API_URL:-http://127.0.0.1:1/v1}}" \
        --judge_model "${JUDGE_MODEL:-${JUDGE_MODEL_ID:-qwen3-30b-a3b-instruct-2507}}"
      )
      if [ "$EVAL_STAGE" = "predict" ]; then
        CORPUSQA_CMD+=(--skip_eval)
      fi
      run_step corpusqa_128k_smoke "${CORPUSQA_CMD[@]}"
      if [ "$EVAL_STAGE" = "predict" ]; then
        require_jsonl_count corpusqa_128k_smoke "$CORPUSQA_DATA" "$OUT/runs/${MODEL_NAME}_corpusqa_128k.jsonl" || true
      fi
    else
      echo "[corpusqa] skip missing $CORPUSQA_SRC"
      echo 99 > "$OUT/corpusqa_128k_smoke.exit"
    fi
  fi
fi

if want_bench lbv1qa; then
	  LBV1_SRC="${LBV1_DATA_DIR:-$DATA_ROOT/LongBench/Longbench/data}"
	  LBV1_DATA="$OUT/data/lbv1qa"
	  LBV1_EVAL="$OUT/data/lbv1qa_eval_maxout.py"
	  mkdir -p "$LBV1_DATA"
	  python - "$BENCH_ROOT/lbv1qa_eval.py" "$LBV1_EVAL" "$MAX_OUTPUT_TOKENS" <<'PYLBV1PATCH'
import sys
from pathlib import Path

src, dst, max_tokens = sys.argv[1:4]
text = Path(src).read_text(encoding="utf-8")
old = "temperature=0.7, top_p=0.95, max_tokens=51200)"
new = f"temperature=0.7, top_p=0.95, max_tokens={int(max_tokens)})"
if old not in text:
    raise SystemExit(f"Could not patch LBV1 max_tokens in {src}")
Path(dst).write_text(text.replace(old, new), encoding="utf-8")
PYLBV1PATCH
	  python - "$LBV1_SRC" "$LBV1_DATA" "$SMOKE_LIMIT" "$MAX_INPUT_TOKENS" "$EFFECTIVE_MODEL_PATH" <<'PYLBV1DATA'
import json
import os
import sys

from transformers import AutoTokenizer

src_dir, dst_dir, limit_text, max_tokens_text, model_path = sys.argv[1:6]
limit = int(limit_text)
max_tokens = int(max_tokens_text)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
subsets = ["narrativeqa", "qasper", "hotpotqa", "2wikimqa", "musique"]
os.makedirs(dst_dir, exist_ok=True)
missing = []

PROMPT_TEMPLATE = """Please read the following text and answer the question below.
<text>
{context}
</text>
{question}
Format your response as follows: "Therefore, the answer is (insert answer here)"."""

def chat_token_len(prompt):
    messages = [{"role": "user", "content": prompt}]
    return len(tok.apply_chat_template(messages, add_generation_prompt=True))

def truncate_context(context, question):
    def render(ctx):
        return PROMPT_TEMPLATE.format(context=ctx, question=question)
    if max_tokens <= 0 or chat_token_len(render(context)) <= max_tokens:
        return context
    context_ids = tok.encode(context, add_special_tokens=False)
    lo, hi = 0, len(context_ids)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        left = mid // 2
        right = mid - left
        candidate = tok.decode(context_ids[:left] + context_ids[-right:], skip_special_tokens=True)
        if chat_token_len(render(candidate)) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best

for subset in subsets:
    src_path = os.path.join(src_dir, f"{subset}.jsonl")
    dst_path = os.path.join(dst_dir, f"{subset}.jsonl")
    if not os.path.exists(src_path):
        missing.append(src_path)
        continue
    with open(src_path, encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if limit > 0 and i >= limit:
                break
            item = json.loads(line)
            context = truncate_context(item["context"], item["input"])
            record = {
                "id": f"{subset}_{i}",
                "dataset": subset,
                "input": item["input"],
                "context": context,
                "answers": item["answers"],
                "length": item.get("length", 0),
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

if missing:
    print("[lbv1qa] missing subsets:")
    for path in missing:
        print(path)
PYLBV1DATA
	  print_step_progress lbv1qa_smoke \
	    "$(jsonl_tree_line_count "$LBV1_DATA")" \
	    "$(jsonl_tree_line_count "$OUT/runs/lbv1qa")" \
	    "$LBV1_DATA" \
	    "$OUT/runs/lbv1qa"
	  LBV1_CMD=(
	    python "$LBV1_EVAL"
    --model "$MODEL_NAME"
    --api_url "http://127.0.0.1:${PORT}/v1"
    --api_key EMPTY
    --concurrency 4
    --data_dir "$LBV1_DATA"
    --runs_dir "$OUT/runs/lbv1qa"
    --evals_dir "$OUT/evals/lbv1qa"
    --max_context_chars "${LBV1_MAX_CONTEXT_CHARS:-500000}"
    --skip_download
    --judge_base_url "${JUDGE_BASE_URL:-${JUDGE_API_URL:-http://127.0.0.1:1/v1}}"
    --judge_model "${JUDGE_MODEL:-${JUDGE_MODEL_ID:-qwen3-30b-a3b-instruct-2507}}"
  )
  if [ "$EVAL_STAGE" = "predict" ]; then
    LBV1_CMD+=(--skip_eval)
  elif [ -n "$JUDGE_API_KEY" ]; then
    LBV1_CMD+=(--judge_api_key "$JUDGE_API_KEY")
  fi
  run_step lbv1qa_smoke "${LBV1_CMD[@]}"
  if [ "$EVAL_STAGE" = "predict" ]; then
    require_jsonl_tree_count lbv1qa_smoke "$LBV1_DATA" "$OUT/runs/lbv1qa" || true
  fi
fi

cleanup
trap - EXIT

echo "===== SUMMARY exits ====="
bad=0
for f in "$OUT"/*.exit; do
  [ -f "$f" ] || continue
  code="$(cat "$f")"
  echo "$(basename "$f" .exit): ${code}"
  if [ "$code" != "0" ] && [ "$code" != "99" ]; then
    bad=1
  fi
done
find "$OUT" -maxdepth 5 -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.log" -o -name "*.exit" -o -name "*.yaml" -o -name "*.env" \) | sort | sed -n '1,260p'
echo "[job] done out=$OUT"
exit "$bad"
