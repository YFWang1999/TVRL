#!/usr/bin/env bash
# Matched scalar-reward GRPO baseline: the same Qwen3.5-9B reward, prompts, and
# sampler as scripts/train_tvrl.sh, with the advantage broadcast uniformly over
# video tokens instead of routed by token credit.
set -euo pipefail
RUN_NAME="${RUN_NAME:-grpo-qwen35-9b-sage-uniform}" \
RATIO_MODE=uniform_weighted_logprob \
bash "$(dirname "${BASH_SOURCE[0]}")/train_tvrl.sh"
