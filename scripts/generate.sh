#!/usr/bin/env bash
# Generate evaluation videos from the base model or a post-trained checkpoint.
#
#   bash scripts/generate.sh                                   # base model
#   CHECKPOINT_PATH=outputs/<run>/checkpoints/<step>/transformer bash scripts/generate.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME:-base_model}"
MODEL_PATH="${MODEL_PATH:-./ckpts}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
VALID_VIDEO_CSV="${VALID_VIDEO_CSV:-assets/val_prompts.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-./outputs/${RUN_NAME}/samples}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
SEED="${SEED:-42}"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

torchrun --nproc_per_node="${NUM_GPUS}" generate.py \
  --valid_video_csv "${VALID_VIDEO_CSV}" \
  --resolution 480p \
  --model_path "${MODEL_PATH}" \
  --aspect_ratio 16:9 \
  --fixed_size 480x864 \
  --num_inference_steps 40 \
  --video_length 121 \
  --negative_prompt "" \
  --seed "${SEED}" \
  --sr false \
  --rewrite false \
  --dtype bf16 \
  --offloading false \
  --overlap_group_offloading false \
  --cfg_distilled false \
  --enable_step_distill false \
  --sparse_attn false \
  --use_sageattn false \
  --enable_cache false \
  --cache_type deepcache \
  --output_path "${OUTPUT_PATH}" \
  "${CHECKPOINT_ARGS[@]}"
