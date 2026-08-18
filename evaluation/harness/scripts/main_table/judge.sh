#!/usr/bin/env bash
set -euo pipefail

: "${PUBLIC_EVAL_ROOT:?PUBLIC_EVAL_ROOT must point to the evaluation directory}"
: "${PRED_OUT:?PRED_OUT is required}"
: "${JUDGE_MODEL_PATH:?JUDGE_MODEL_PATH is required}"
: "${JUDGE_OUT_ROOT:?JUDGE_OUT_ROOT is required}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
python() {
  "$PYTHON_BIN" "$@"
}
export -f python

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

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

export BENCH_ROOT="$PUBLIC_EVAL_ROOT/harness/benchmarks"
export PRED_MODEL_NAME="${PRED_MODEL_NAME:-main_table_candidate}"
export JUDGE_MODEL_NAME=qwen3-30b-a3b-instruct-2507-local
export JUDGE_PORT="${JUDGE_PORT:-$(pick_free_port 24000 29999)}"
export JUDGE_TP="${JUDGE_TP:-4}"
export JUDGE_MAX_MODEL_LEN=32768
export JUDGE_MAX_TOKENS=2048
export JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
export JUDGE_VISIBLE_DEVICES="${JUDGE_VISIBLE_DEVICES:-0,1,2,3}"
export JUDGE_QWEN3_THINKING_MODE=auto
export JUDGE_TARGETS=corpusqa,lbv1qa
export RESUME_MODE=auto
export OUT_ROOT="$JUDGE_OUT_ROOT"
export RUN_ID="${RUN_ID:-${RESUME_ID:-}}"

if [ -z "${OUT:-}" ]; then
  case "$RESUME_MODE" in
    auto|force|true|1|yes)
      if [ -z "$RUN_ID" ]; then
        RUN_ID="$(python - "$PRED_MODEL_NAME" "$JUDGE_MODEL_NAME" "$JUDGE_TARGETS" "$JUDGE_MAX_MODEL_LEN" "$JUDGE_MAX_TOKENS" <<'PYID'
import re
import sys

pred, judge, targets, max_len, max_tokens = sys.argv[1:6]
raw = f"{pred}-judge-{judge}-{targets}-len{max_len}_tok{max_tokens}"
safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", raw).strip("._-")
print(safe[:180] or "main_table_judge_resume_run")
PYID
)"
      fi
      OUT="$OUT_ROOT/$RUN_ID"
      ;;
    *)
      OUT="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)-${PRED_MODEL_NAME}_judge_qwen3_30b"
      ;;
  esac
fi
export OUT RUN_ID
export ATTEMPT_ID="${ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)-pid$$}"

resume_enabled() {
  case "$RESUME_MODE" in
    disable|disabled|false|0|no) return 1 ;;
    *) return 0 ;;
  esac
}

want_judge_target() {
  case ",${JUDGE_TARGETS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

step_exit_success() {
  local name="$1"
  resume_enabled || return 1
  [ -f "$OUT/${name}.exit" ] || return 1
  [ "$(cat "$OUT/${name}.exit" 2>/dev/null)" = "0" ]
}

requested_judge_steps_complete() {
  local any=0
  if want_judge_target corpusqa; then
    any=1
    step_exit_success corpusqa_judge || return 1
  fi
  if want_judge_target lbv1qa; then
    any=1
    step_exit_success lbv1qa_judge || return 1
  fi
  [ "$any" = "1" ]
}

print_exit_summary() {
  echo "===== SUMMARY exits ====="
  for f in "$OUT"/*.exit; do
    [ -f "$f" ] || continue
    echo "$(basename "$f" .exit): $(cat "$f")"
  done
}

mkdir -p "$OUT" "$OUT/evals" "$OUT/logs" "$OUT/attempts"
exec > >(tee -a "$OUT/job.log" "$OUT/attempts/${ATTEMPT_ID}.log") 2>&1

echo "[judge] out=$OUT"
echo "[judge] out_root=$OUT_ROOT run_id=${RUN_ID:-} resume_mode=$RESUME_MODE attempt_id=$ATTEMPT_ID targets=$JUDGE_TARGETS"
echo "[judge] pred_out=$PRED_OUT"
echo "[judge] pred_model=$PRED_MODEL_NAME"
echo "[judge] judge_model_path=$JUDGE_MODEL_PATH"
echo "[judge] judge_model_name=$JUDGE_MODEL_NAME"
echo "[judge] python=$(which python)"
echo "[judge] python_version=$(python --version 2>&1)"
echo "[judge] port=$JUDGE_PORT tp=$JUDGE_TP max_model_len=$JUDGE_MAX_MODEL_LEN max_tokens=$JUDGE_MAX_TOKENS"
python scripts/main_table/write_run_config.py --kind judge --phase initial

if requested_judge_steps_complete; then
  echo "[resume] all requested judge steps already have successful exits; exiting without starting judge vLLM"
  print_exit_summary
  python "$PUBLIC_EVAL_ROOT/aggregate_main_table.py" \
    --pred-out "$PRED_OUT" \
    --model-name "$PRED_MODEL_NAME" \
    --judge-out "$OUT" \
    --out "$OUT/main_table_scores.json" \
    --csv "$OUT/main_table_scores.csv"
  echo "[judge] done out=$OUT"
  exit 0
fi

test -d "$BENCH_ROOT"
test -d "$PRED_OUT"
test -d "$JUDGE_MODEL_PATH"
if want_judge_target corpusqa; then
  test -f "$PRED_OUT/data/corpusqa_128k_smoke.jsonl"
  test -f "$PRED_OUT/runs/${PRED_MODEL_NAME}_corpusqa_128k.jsonl"
fi
if want_judge_target lbv1qa; then
  test -d "$PRED_OUT/data/lbv1qa"
  test -d "$PRED_OUT/runs/lbv1qa"
fi

patch_judge_scripts() {
  python - "$BENCH_ROOT/corpusqa_eval.py" "$OUT/corpusqa_eval_judge.py" \
           "$BENCH_ROOT/lbv1qa_eval.py" "$OUT/lbv1qa_eval_judge.py" <<'PYPATCH'
import sys
from pathlib import Path

corpus_src, corpus_dst, lbv_src, lbv_dst = map(Path, sys.argv[1:5])

corpus = corpus_src.read_text(encoding="utf-8")
if "import os\n" not in corpus:
    corpus = "import os\n" + corpus
old = """resp = client.chat.completions.create(
                    model=args.judge_model, messages=msgs, temperature=0.0)"""
new = """resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=msgs,
                    temperature=0.0,
                    max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
                )"""
if old not in corpus:
    raise SystemExit("Could not patch corpusqa judge max_tokens")
corpus_dst.write_text(corpus.replace(old, new), encoding="utf-8")

lbv = lbv_src.read_text(encoding="utf-8")
if "import os\n" not in lbv:
    lbv = "import os\n" + lbv
old = """response = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )"""
new = """response = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
            )"""
if old not in lbv:
    raise SystemExit("Could not patch lbv1qa judge max_tokens")
lbv_dst.write_text(lbv.replace(old, new), encoding="utf-8")
PYPATCH
}

wait_chat_server() {
  for _ in $(seq 1 240); do
    if curl -s -o /dev/null -w "%{http_code}" \
      -X POST "http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$JUDGE_MODEL_NAME"'","messages":[{"role":"user","content":"Answer with [[YES]]."}],"max_tokens":16,"temperature":0}' | grep -q 200; then
      echo "[vllm] judge ready on port $JUDGE_PORT"
      return 0
    fi
    sleep 5
  done
  echo "[vllm] judge startup timeout on port $JUDGE_PORT"
  return 1
}

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    if [ "${VLLM_OWN_SESSION:-0}" = "1" ]; then
      kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
    fi
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 2
    if [ "${VLLM_OWN_SESSION:-0}" = "1" ]; then
      kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
    fi
    kill -9 "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}
trap cleanup EXIT

patch_judge_scripts

JUDGE_VLLM_CHAT_TEMPLATE_ARGS=()
case "$(printf '%s' "$JUDGE_QWEN3_THINKING_MODE" | tr '[:upper:]' '[:lower:]' | tr '-' '_')" in
  auto|"")
    JUDGE_QWEN3_THINKING_MODE_NORM=auto
    ;;
  think|thinking|enable|enabled|true|1|yes)
    JUDGE_QWEN3_THINKING_MODE_NORM=think
    JUDGE_VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
    ;;
  nothink|no_think|non_thinking|disable|disabled|false|0|no)
    JUDGE_QWEN3_THINKING_MODE_NORM=nothink
    JUDGE_VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
    ;;
  *)
    echo "[vllm] invalid JUDGE_QWEN3_THINKING_MODE=$JUDGE_QWEN3_THINKING_MODE; expected auto, think, or nothink"
    exit 2
    ;;
esac

if [ -n "${JUDGE_VLLM_CHAT_TEMPLATE_KWARGS:-}" ]; then
  if python -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- "--default-chat-template-kwargs"; then
    JUDGE_VLLM_CHAT_TEMPLATE_ARGS=(--default-chat-template-kwargs "$JUDGE_VLLM_CHAT_TEMPLATE_KWARGS")
    export JUDGE_VLLM_CHAT_TEMPLATE_KWARGS
    echo "[vllm] judge_default_chat_template_kwargs=$JUDGE_VLLM_CHAT_TEMPLATE_KWARGS"
  elif [ "$JUDGE_QWEN3_THINKING_MODE_NORM" = "nothink" ]; then
    JUDGE_MODEL_PATH="$(python scripts/main_table/patch_qwen3_chat_template.py \
      --model-path "$JUDGE_MODEL_PATH" \
      --out "$OUT" \
      --mode "$JUDGE_QWEN3_THINKING_MODE_NORM")"
    echo "[vllm] default-chat-template-kwargs unsupported; using patched judge no-think model overlay: $JUDGE_MODEL_PATH"
  else
    echo "[vllm] default-chat-template-kwargs unsupported; JUDGE_QWEN3_THINKING_MODE=think is equivalent to Qwen3 default for this template"
  fi
fi

echo "[judge] qwen3_thinking_mode=$JUDGE_QWEN3_THINKING_MODE"
python scripts/main_table/write_run_config.py --kind judge --phase resolved

JUDGE_VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$JUDGE_MODEL_PATH" \
  --served-model-name "$JUDGE_MODEL_NAME" \
  --tensor-parallel-size "$JUDGE_TP" \
  --max-model-len "$JUDGE_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --port "$JUDGE_PORT" \
  "${JUDGE_VLLM_CHAT_TEMPLATE_ARGS[@]}"
)
if command -v setsid >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$JUDGE_VISIBLE_DEVICES" setsid "${JUDGE_VLLM_CMD[@]}" > "$OUT/vllm.log" 2>&1 &
  VLLM_OWN_SESSION=1
else
  CUDA_VISIBLE_DEVICES="$JUDGE_VISIBLE_DEVICES" "${JUDGE_VLLM_CMD[@]}" > "$OUT/vllm.log" 2>&1 &
  VLLM_OWN_SESSION=0
fi
VLLM_PID=$!

echo "[vllm] pid=$VLLM_PID waiting..."
if ! wait_chat_server; then
  tail -260 "$OUT/vllm.log" || true
  exit 1
fi

run_step() {
  name="$1"; shift
  if step_exit_success "$name"; then
    echo "===== STEP ${name} SKIP completed exit=0 ====="
    return 0
  fi
  echo "===== STEP ${name} START ====="
  set +e
  ( "$@" ) > "$OUT/${name}.log" 2>&1
  code=$?
  set -e
  echo "===== STEP ${name} EXIT ${code} ====="
  tail -180 "$OUT/${name}.log" || true
  echo "$code" > "$OUT/${name}.exit"
}

if want_judge_target corpusqa; then
  run_step corpusqa_judge python "$OUT/corpusqa_eval_judge.py" \
    --data_file "$PRED_OUT/data/corpusqa_128k_smoke.jsonl" \
    --model "$PRED_MODEL_NAME" \
    --infer_output "$PRED_OUT/runs/${PRED_MODEL_NAME}_corpusqa_128k.jsonl" \
    --eval_output "$OUT/evals/corpusqa_128k_eval.jsonl" \
    --judge_api_key EMPTY \
    --judge_base_url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge_model "$JUDGE_MODEL_NAME" \
    --skip_infer
fi

if want_judge_target lbv1qa; then
  run_step lbv1qa_judge python "$OUT/lbv1qa_eval_judge.py" \
    --model "$PRED_MODEL_NAME" \
    --data_dir "$PRED_OUT/data/lbv1qa" \
    --runs_dir "$PRED_OUT/runs/lbv1qa" \
    --evals_dir "$OUT/evals/lbv1qa" \
    --judge_api_key EMPTY \
    --judge_base_url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --judge_model "$JUDGE_MODEL_NAME" \
    --skip_download \
    --skip_infer
fi

python - "$OUT" "$PRED_MODEL_NAME" <<'PYSUM'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
model = sys.argv[2]

def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

corpus_eval = load_jsonl(out / "evals" / "corpusqa_128k_eval.jsonl")
corpus_correct = sum(1 for item in corpus_eval if item.get("correct"))
summary = {
    "corpusqa": {
        "num": len(corpus_eval),
        "correct": corpus_correct,
        "score": (corpus_correct / len(corpus_eval) * 100) if corpus_eval else None,
    },
    "lbv1qa": {},
}

lbv_summary = out / "evals" / "lbv1qa" / f"{model}_lbv1qa_summary.json"
if lbv_summary.exists():
    scores = json.loads(lbv_summary.read_text(encoding="utf-8"))
    required = ["narrativeqa", "qasper", "hotpotqa", "2wikimqa", "musique"]
    missing = [name for name in required if name not in scores]
    summary["lbv1qa"]["subsets"] = scores
    summary["lbv1qa"]["missing_subsets"] = missing
    summary["lbv1qa"]["overall"] = (
        sum(scores[name] for name in required) / len(required) if not missing else None
    )
    single = [scores[k] for k in ["narrativeqa", "qasper"] if k in scores]
    multi = [scores[k] for k in ["hotpotqa", "2wikimqa", "musique"] if k in scores]
    summary["lbv1qa"]["single_doc_avg"] = (sum(single) / len(single)) if single else None
    summary["lbv1qa"]["multi_doc_avg"] = (sum(multi) / len(multi)) if multi else None

(out / "judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PYSUM

python "$PUBLIC_EVAL_ROOT/aggregate_main_table.py" \
  --pred-out "$PRED_OUT" \
  --model-name "$PRED_MODEL_NAME" \
  --judge-out "$OUT" \
  --out "$OUT/main_table_scores.json" \
  --csv "$OUT/main_table_scores.csv"

echo "===== SUMMARY exits ====="
bad=0
for f in "$OUT"/*.exit; do
  [ -f "$f" ] || continue
  code="$(cat "$f")"
  echo "$(basename "$f" .exit): ${code}"
  if [ "$code" != "0" ]; then
    bad=1
  fi
done
find "$OUT" -maxdepth 5 -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.log" -o -name "*.exit" -o -name "*.yaml" -o -name "*.env" \) | sort | sed -n '1,260p'
exit "$bad"
