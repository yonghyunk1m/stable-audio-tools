#!/usr/bin/env bash
set -euo pipefail
#
# Master launcher for ISMIR experiments on FMA-Large.
# Usage:
#   ./scripts/run_experiments.sh <case>
#
# Cases:
#   case2      - SFT baseline: all FMA, no score conditioning
#   case3a     - Score cond: input_add adapter only (prepend global mode)
#   case3b     - Score cond: adaLN only (scale/shift/gate per block)
#   case3c     - Score cond: hybrid (adaLN + input_add adapter)
#   case4      - Filtered FMA: reward-filtered subset, no score conditioning
#
# All runs use GPU 8,9 by default.  Override: CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_experiments.sh case3b
#

CASE="${1:-}"
if [[ -z "$CASE" || "$CASE" == "-h" || "$CASE" == "--help" ]]; then
  head -17 "$0" | tail -15
  echo ""
  echo "Available cases: case2 case3a case3b case3c case4"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_finetune_case3.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-8,9}"

# Shared paths
CKPT_DIR="./checkpoints/sao_small"
CFG_DIR="./configs"

case "$CASE" in
  case2)
    # Case 2: SFT baseline — all FMA, no score conditioning
    # Unfreezes to_global_embed only (seconds_total projection)
    export MODEL_CONFIG="${CKPT_DIR}/model_config.json"
    export DATASET_CONFIG="${CFG_DIR}/dataset_fma_all.json"
    export VAL_DATASET_CONFIG="${CFG_DIR}/dataset_fma_all.json"
    export SA_UNFREEZE_PROFILE="global"
    export SAVE_DIR_BASE="./results/sao_small_case2"
    export RUN_NAME="case2_sft_all_fma_$(date +%Y%m%d_%H%M%S)"
    ;;
  case3a)
    # Case 3a: Score conditioning via input_add adapter (prepend global mode)
    export MODEL_CONFIG="${CKPT_DIR}/model_config_with_score.json"
    export DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export VAL_DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export SA_UNFREEZE_PROFILE="adapter"
    export SAVE_DIR_BASE="./results/sao_small_case3"
    export RUN_NAME="case3a_adapter_$(date +%Y%m%d_%H%M%S)"
    ;;
  case3b)
    # Case 3b: Score conditioning via adaLN (scale/shift/gate per transformer block)
    export MODEL_CONFIG="${CKPT_DIR}/model_config_with_score_adaln.json"
    export DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export VAL_DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export SA_UNFREEZE_PROFILE="adaln"
    export SAVE_DIR_BASE="./results/sao_small_case3"
    export RUN_NAME="case3b_adaln_$(date +%Y%m%d_%H%M%S)"
    ;;
  case3c)
    # Case 3c: Score conditioning via hybrid (adaLN + input_add adapter)
    export MODEL_CONFIG="${CKPT_DIR}/model_config_with_score_adaln.json"
    export DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export VAL_DATASET_CONFIG="${CFG_DIR}/dataset_fma_scored.json"
    export SA_UNFREEZE_PROFILE="hybrid"
    export SAVE_DIR_BASE="./results/sao_small_case3"
    export RUN_NAME="case3c_hybrid_$(date +%Y%m%d_%H%M%S)"
    ;;
  case4)
    # Case 4: Reward-filtered FMA subset, no score conditioning
    export MODEL_CONFIG="${CKPT_DIR}/model_config.json"
    export DATASET_CONFIG="${CFG_DIR}/dataset_fma_filtered.json"
    export VAL_DATASET_CONFIG="${CFG_DIR}/dataset_fma_filtered.json"
    export SA_UNFREEZE_PROFILE="global"
    export SAVE_DIR_BASE="./results/sao_small_case4"
    export RUN_NAME="case4_filtered_fma_$(date +%Y%m%d_%H%M%S)"
    ;;
  *)
    echo "[ERROR] Unknown case: $CASE"
    echo "Available cases: case2 case3a case3b case3c case4"
    exit 1
    ;;
esac

echo "=========================================="
echo " Launching experiment: $CASE"
echo " Model config: ${MODEL_CONFIG}"
echo " Dataset config: ${DATASET_CONFIG}"
echo " Unfreeze profile: ${SA_UNFREEZE_PROFILE}"
echo " Run name: ${RUN_NAME}"
echo " GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

shift  # remove $CASE, pass remaining args to launcher
exec "${LAUNCHER}" "$@"
