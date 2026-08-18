#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/gc-opd-entrypoints.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

touch "${TMP_ROOT}/train.parquet" "${TMP_ROOT}/val.parquet"
mkdir -p "${TMP_ROOT}/student-4b" "${TMP_ROOT}/student-8b"
mkdir -p "${TMP_ROOT}/teacher" "${TMP_ROOT}/output"
mkdir -p "${TMP_ROOT}/default-data/golongrl_32k"
mkdir -p "${TMP_ROOT}/default-models/Qwen3-4B"
mkdir -p "${TMP_ROOT}/default-models/Qwen3-8B"
mkdir -p "${TMP_ROOT}/default-models/Qwen3-30B-A3B-Thinking-2507"
touch "${TMP_ROOT}/default-data/golongrl_32k/train.parquet"
touch "${TMP_ROOT}/default-data/golongrl_32k/val.parquet"

run_dry() {
    local launcher=$1
    shift
    env \
        TRAIN_DATA="${TMP_ROOT}/train.parquet" \
        VAL_DATA="${TMP_ROOT}/val.parquet" \
        STUDENT_MODEL_4B="${TMP_ROOT}/student-4b" \
        STUDENT_MODEL_8B="${TMP_ROOT}/student-8b" \
        TEACHER_MODEL="${TMP_ROOT}/teacher" \
        OUTPUT_DIR="${TMP_ROOT}/output" \
        RUN_NAME=entrypoint_test \
        DRY_RUN=1 \
        TRAIN_BATCH_SIZE=99 \
        GC_OPD_RESIDUAL_BETA=9.0 \
        GC_OPD_CREDIT_MODE=uniform \
        OPD_ADV_CLIP=3.0 \
        bash "${ROOT_DIR}/scripts/${launcher}" "$@"
}

gc_4b_cmd=$(run_dry run_gc_opd_4b_training.sh)
gc_8b_cmd=$(run_dry run_gc_opd_8b_training.sh)
opd_4b_cmd=$(run_dry run_opd_4b_training.sh)
opd_8b_cmd=$(run_dry run_opd_8b_training.sh)

default_4b_cmd=$(
    env \
        TRAIN_DATA= \
        VAL_DATA= \
        STUDENT_MODEL_4B= \
        STUDENT_MODEL_8B= \
        TEACHER_MODEL= \
        DATA_DIR="${TMP_ROOT}/default-data" \
        MODEL_DIR="${TMP_ROOT}/default-models" \
        OUTPUT_DIR="${TMP_ROOT}/default-output" \
        RUN_NAME=default_path_test \
        DRY_RUN=1 \
        bash "${ROOT_DIR}/scripts/run_gc_opd_4b_training.sh"
)

[[ "${default_4b_cmd}" == *"data.train_files=${TMP_ROOT}/default-data/golongrl_32k/train.parquet"* ]]
[[ "${default_4b_cmd}" == *"data.val_files=\\['${TMP_ROOT}/default-data/golongrl_32k/val.parquet'\\]"* ]]
[[ "${default_4b_cmd}" == *"actor_rollout_ref.model.path=${TMP_ROOT}/default-models/Qwen3-4B"* ]]
[[ "${default_4b_cmd}" == *"actor_rollout_ref.ref.model.path=${TMP_ROOT}/default-models/Qwen3-30B-A3B-Thinking-2507"* ]]
[[ "${default_4b_cmd}" == *"trainer.default_local_dir=${TMP_ROOT}/default-output/checkpoints/default_path_test"* ]]

for command in "${gc_4b_cmd}" "${gc_8b_cmd}" "${opd_4b_cmd}" "${opd_8b_cmd}"; do
    [[ "${command}" == *"data.train_batch_size=32"* ]]
    [[ "${command}" != *"data.train_batch_size=99"* ]]
    [[ "${command}" == *"actor_rollout_ref.rollout.n=8"* ]]
    [[ "${command}" == *"data.max_prompt_length=32768"* ]]
    [[ "${command}" == *"data.max_response_length=10240"* ]]
    [[ "${command}" == *"trainer.total_training_steps=100"* ]]
    [[ "${command}" == *"trainer.val_before_train=False"* ]]
    [[ "${command}" == *"enable_thinking=False"* ]]
    [[ "${command}" == *"actor_rollout_ref.ref.model.path=${TMP_ROOT}/teacher"* ]]
done

val_before_cmd=$(VAL_BEFORE_TRAIN=True run_dry run_gc_opd_4b_training.sh)
[[ "${val_before_cmd}" == *"trainer.val_before_train=True"* ]]

if VAL_BEFORE_TRAIN=invalid run_dry run_gc_opd_4b_training.sh >/dev/null 2>&1; then
    echo "Entrypoint unexpectedly accepted an invalid VAL_BEFORE_TRAIN value." >&2
    exit 1
fi

for command in "${gc_4b_cmd}" "${gc_8b_cmd}"; do
    [[ "${command}" == *"policy_loss.method=gc_opd"* ]]
    [[ "${command}" == *"gc_opd_credit_mode=raca"* ]]
    [[ "${command}" != *"gc_opd_credit_mode=uniform"* ]]
    [[ "${command}" == *"gc_opd_residual_beta=0.10"* ]]
    [[ "${command}" != *"gc_opd_residual_beta=9.0"* ]]
done

for command in "${opd_4b_cmd}" "${opd_8b_cmd}"; do
    [[ "${command}" == *"policy_loss.method=gc_opd"* ]]
    [[ "${command}" == *"only_reverse_kl_advantages=False"* ]]
    [[ "${command}" == *"gc_opd_credit_mode=raca"* ]]
    [[ "${command}" == *"gc_opd_residual_beta=0.0"* ]]
    [[ "${command}" != *"gc_opd_residual_beta=0.10"* ]]
    [[ "${command}" != *"gc_opd_residual_beta=9.0"* ]]
    [[ "${command}" != *"policy_loss.opd_adv_clip="* ]]
done

[[ "${gc_4b_cmd}" == *"actor_rollout_ref.model.path=${TMP_ROOT}/student-4b"* ]]
[[ "${gc_8b_cmd}" == *"actor_rollout_ref.model.path=${TMP_ROOT}/student-8b"* ]]
[[ "${opd_4b_cmd}" == *"actor_rollout_ref.model.path=${TMP_ROOT}/student-4b"* ]]
[[ "${opd_4b_cmd}" == *"policy_loss.gc_opd_adv_clip=0.0"* ]]
[[ "${opd_8b_cmd}" == *"actor_rollout_ref.model.path=${TMP_ROOT}/student-8b"* ]]
[[ "${opd_8b_cmd}" == *"policy_loss.gc_opd_adv_clip=10.0"* ]]
[[ "${opd_8b_cmd}" != *"policy_loss.gc_opd_adv_clip=3.0"* ]]

if run_dry run_gc_opd_8b_training.sh trainer.total_training_steps=1 >/dev/null 2>&1; then
    echo "Entrypoint unexpectedly accepted a training override." >&2
    exit 1
fi

echo "Fixed GC-OPD and OPD 4B/8B main-table entrypoints passed dry-run checks."
