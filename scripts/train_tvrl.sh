#!/usr/bin/env bash
# TVRL post-training of HunyuanVideo-1.5: Qwen3.5-9B answer likelihoods as the
# reward, 3x3-window reward-sensitivity maps as token credit, SAGE sampler.
#
#   bash scripts/train_tvrl.sh
#
# Override any variable below from the environment, e.g.
#   SDE_TYPE=flow_grpo CREDIT_WINDOW=7 RUN_NAME=tvrl-flow-7x7 bash scripts/train_tvrl.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME:-tvrl-qwen35-9b-sage-3x3}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
PRETRAINED_MODEL_ROOT="${PRETRAINED_MODEL_ROOT:-./ckpts}"
TRAIN_VIDEO_CSV="${TRAIN_VIDEO_CSV:-assets/train_prompts_binary5.csv}"
VLM_REWARD_MODEL_PATH="${VLM_REWARD_MODEL_PATH:-./ckpts/Qwen3.5-9B}"
VLM_REWARD_MODEL_FAMILY="${VLM_REWARD_MODEL_FAMILY:-qwen3_5}"
SDE_TYPE="${SDE_TYPE:-sage_grpo}"              # sage_grpo | flow_grpo | dance_grpo
# Token credit: "window" smooths the 16x16 routing grid with a CREDIT_WINDOW x
# CREDIT_WINDOW average (3 = default, 7 = coarser, 1 = unsmoothed cells);
# CREDIT_GRANULARITY=frame gives one weight per frame.
CREDIT_GRANULARITY="${CREDIT_GRANULARITY:-window}"
CREDIT_BINS="${CREDIT_BINS:-16}"
CREDIT_WINDOW="${CREDIT_WINDOW:-3}"
# weighted_logprob = TVRL routing; uniform_weighted_logprob = scalar GRPO with the same reward.
RATIO_MODE="${RATIO_MODE:-weighted_logprob}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/${RUN_NAME}/checkpoints}"
ENABLE_WANDB="${ENABLE_WANDB:-False}"           # set True and export WANDB_API_KEY to log

LOG_DIR="logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" post_train.py \
  --pretrained_model_root "${PRETRAINED_MODEL_ROOT}" \
  --train_video_csv "${TRAIN_VIDEO_CSV}" \
  --reward_model vlm_reward \
  --vlm_reward_model_path "${VLM_REWARD_MODEL_PATH}" \
  --vlm_reward_model_family "${VLM_REWARD_MODEL_FAMILY}" \
  --vlm_reward_num_frames 20 \
  --vlm_reward_max_pixels 50176 \
  --vlm_reward_max_new_tokens 128 \
  --vlm_reward_batch_size 1 \
  --vlm_reward_score_type token_credit \
  --vlm_reward_token_credit_max_questions 5 \
  --token_credit_loss_mode video_adv \
  --token_credit_ratio_mode "${RATIO_MODE}" \
  --token_credit_spatial_granularity "${CREDIT_GRANULARITY}" \
  --token_credit_spatial_bins "${CREDIT_BINS}" \
  --token_credit_spatial_window "${CREDIT_WINDOW}" \
  --token_credit_chunk_size 1 \
  --learning_rate 1e-5 \
  --batch_size 2 \
  --num_generations 4 \
  --max_steps "${MAX_STEPS}" \
  --output_dir ./outputs \
  --enable_wandb "${ENABLE_WANDB}" \
  --project_name "TVRL" \
  --run_name "${RUN_NAME}" \
  --enable_fsdp \
  --enable_gradient_checkpointing \
  --sp_size 2 \
  --reward_checkpoint_mode "v3" \
  --validation_interval 0 \
  --validate_at_step0 False \
  --validate_video_length 121 \
  --validation_timestep_shift 5.0 \
  --use_grad_balancing True \
  --enable_timestep_permutation True \
  --sde_type "${SDE_TYPE}" \
  --kl_weight 1e-5 \
  --kl_coef 1e-7 \
  --use_moving_KL True \
  --update_ref_model_step 10 \
  --use_dual_kl True \
  --dual_kl_moving_weight 1.0 \
  --dual_kl_step_weight 0.1 \
  --reference_mode_offload True \
  --save_interval "${SAVE_INTERVAL}" \
  --checkpoints_directory "${CHECKPOINTS_DIR}" \
  2>&1 | tee -a "${LOG_DIR}/train.log"
