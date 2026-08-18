#!/usr/bin/env bash
set -euo pipefail

# Run prediction first, then run the local LLM-as-judge stage against the
# prediction artifacts from the same run directory.

: "${OUT_ROOT:?OUT_ROOT is required}"
: "${JUDGE_OUT_ROOT:?JUDGE_OUT_ROOT is required}"

PREDICT_RUN_ID="${RUN_ID:-main_table_five_task}"
PREDICT_OUT="${OUT:-$OUT_ROOT/$PREDICT_RUN_ID}"
export EVAL_STAGE=predict

echo "[two-stage] predict_out=$PREDICT_OUT"
echo "[two-stage] predict_run_id=$PREDICT_RUN_ID"
echo "[two-stage] target_benchmarks=${TARGET_BENCHMARKS:-}"

bash scripts/main_table/predict.sh

export PRED_OUT="${PRED_OUT:-$PREDICT_OUT}"
export PRED_MODEL_NAME="${PRED_MODEL_NAME:-${MODEL_NAME:-main_table_candidate}}"
export JUDGE_TARGETS=corpusqa,lbv1qa
export RESUME_MODE=auto

unset RUN_ID
unset OUT

echo "[two-stage] judge_targets=$JUDGE_TARGETS"
echo "[two-stage] pred_out=$PRED_OUT"
echo "[two-stage] pred_model=$PRED_MODEL_NAME"
echo "[two-stage] judge_out_root=$JUDGE_OUT_ROOT"

bash scripts/main_table/judge.sh
