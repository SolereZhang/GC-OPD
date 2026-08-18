#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_main_table_training.sh {gc_opd|opd} {4b|8b}

Default locations relative to the repository:
  ../data/golongrl_32k/{train,val}.parquet
  ../models/{Qwen3-4B,Qwen3-8B,Qwen3-30B-A3B-Thinking-2507}
  ../outputs

Directory overrides:
  DATA_DIR, MODEL_DIR, OUTPUT_DIR

Individual path overrides:
  TRAIN_DATA, VAL_DATA, STUDENT_MODEL_4B, STUDENT_MODEL_8B, TEACHER_MODEL

Runtime-only options:
  RUN_NAME, CHECKPOINT_DIR, RESUME_MODE, VAL_BEFORE_TRAIN, PYTHON_BIN, DRY_RUN

The main-table training and method settings are fixed in this script.
EOF
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

MODE=$1
SCALE=$2
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATA_DIR=${DATA_DIR:-${ROOT_DIR}/../data}
MODEL_DIR=${MODEL_DIR:-${ROOT_DIR}/../models}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/../outputs}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/golongrl_32k/train.parquet}
VAL_DATA=${VAL_DATA:-${DATA_DIR}/golongrl_32k/val.parquet}
STUDENT_MODEL_4B=${STUDENT_MODEL_4B:-${MODEL_DIR}/Qwen3-4B}
STUDENT_MODEL_8B=${STUDENT_MODEL_8B:-${MODEL_DIR}/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-${MODEL_DIR}/Qwen3-30B-A3B-Thinking-2507}

case "${MODE}" in
    gc_opd|opd) ;;
    *)
        usage
        exit 2
        ;;
esac

case "${SCALE}" in
    4b)
        STUDENT_MODEL=${STUDENT_MODEL_4B}
        ;;
    8b)
        STUDENT_MODEL=${STUDENT_MODEL_8B}
        ;;
    *)
        usage
        exit 2
        ;;
esac

export PYTHONPATH="${ROOT_DIR}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=disabled

for path in "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing data file: ${path}" >&2
        exit 1
    fi
done

for path in "${STUDENT_MODEL}" "${TEACHER_MODEL}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing model path: ${path}" >&2
        exit 1
    fi
done

REWARD_FUNCTION=${ROOT_DIR}/verl/examples/gc_opd/golongrl_reward.py
if [[ ! -f "${REWARD_FUNCTION}" ]]; then
    echo "Missing reward function: ${REWARD_FUNCTION}" >&2
    exit 1
fi

RUN_NAME=${RUN_NAME:-${MODE}_${SCALE}_$(date +%Y%m%d_%H%M%S)}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints/${RUN_NAME}}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
case "${VAL_BEFORE_TRAIN}" in
    True|False) ;;
    *)
        echo "VAL_BEFORE_TRAIN must be True or False, got: ${VAL_BEFORE_TRAIN}" >&2
        exit 2
        ;;
esac
mkdir -p "${CHECKPOINT_DIR}"

METHOD_ARGS=()
case "${MODE}" in
    gc_opd)
        METHOD_ARGS=(
            actor_rollout_ref.actor.policy_loss.method=gc_opd
            actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=False
            actor_rollout_ref.actor.policy_loss.entropy_aware_distill=False
            actor_rollout_ref.actor.policy_loss.gc_opd_teacher_score_agg=mean
            actor_rollout_ref.actor.policy_loss.gc_opd_residual_norm=group_zscore
            actor_rollout_ref.actor.policy_loss.gc_opd_residual_beta=0.10
            actor_rollout_ref.actor.policy_loss.gc_opd_credit_mode=raca
            actor_rollout_ref.actor.policy_loss.gc_opd_credit_cap=5.0
            actor_rollout_ref.actor.policy_loss.gc_opd_group_size=8
            actor_rollout_ref.actor.policy_loss.gc_opd_min_group_std=1e-6
            actor_rollout_ref.actor.policy_loss.gc_opd_min_token_std=1e-6
            actor_rollout_ref.actor.policy_loss.gc_opd_adv_clip=10.0
        )
        ;;
    opd)
        OPD_ADV_CLIP=0.0
        if [[ "${SCALE}" == "8b" ]]; then
            OPD_ADV_CLIP=10.0
        fi
        METHOD_ARGS=(
            actor_rollout_ref.actor.policy_loss.method=gc_opd
            actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=False
            actor_rollout_ref.actor.policy_loss.entropy_aware_distill=False
            actor_rollout_ref.actor.policy_loss.gc_opd_teacher_score_agg=mean
            actor_rollout_ref.actor.policy_loss.gc_opd_residual_norm=group_zscore
            actor_rollout_ref.actor.policy_loss.gc_opd_residual_beta=0.0
            actor_rollout_ref.actor.policy_loss.gc_opd_credit_mode=raca
            actor_rollout_ref.actor.policy_loss.gc_opd_credit_cap=5.0
            actor_rollout_ref.actor.policy_loss.gc_opd_group_size=8
            actor_rollout_ref.actor.policy_loss.gc_opd_min_group_std=1e-6
            actor_rollout_ref.actor.policy_loss.gc_opd_min_token_std=1e-6
            actor_rollout_ref.actor.policy_loss.gc_opd_adv_clip="${OPD_ADV_CLIP}"
        )
        ;;
esac

CMD=(
    "${PYTHON_BIN:-python}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=5.0
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.bypass_mode=false
    data.train_files="${TRAIN_DATA}"
    data.val_files="['${VAL_DATA}']"
    data.train_batch_size=32
    data.train_max_samples=-1
    data.val_max_samples=-1
    data.max_prompt_length=32768
    data.max_response_length=10240
    data.filter_overlong_prompts=False
    data.truncation=error
    data.shuffle=True
    data.seed=42
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    +actor_rollout_ref.model.override_config.max_position_embeddings=43008
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=43008
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.prompt_length=32768
    actor_rollout_ref.rollout.response_length=10240
    actor_rollout_ref.rollout.max_model_len=43008
    actor_rollout_ref.rollout.max_num_batched_tokens=43008
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.n=1
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}"
    +actor_rollout_ref.ref.model.override_config.max_position_embeddings=43008
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=43008
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    +actor_rollout_ref.ref.disable_retokenization=True
    algorithm.use_kl_in_reward=False
    reward_model.reward_manager=naive
    custom_reward_function.path="${REWARD_FUNCTION}"
    custom_reward_function.name=compute_score
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=gc-opd-main-table
    trainer.experiment_name="${RUN_NAME}"
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.save_freq=10
    trainer.test_freq=10
    trainer.total_epochs=1
    trainer.total_training_steps=100
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode="${RESUME_MODE:-disable}"
    "${METHOD_ARGS[@]}"
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"
