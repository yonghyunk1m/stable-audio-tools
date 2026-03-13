#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run_finetune_case3.sh [extra finetune.py args...]

Common overrides (env vars):
  RUN_NAME=...
  CUDA_VISIBLE_DEVICES=8,9
  LOGGER_TYPE=wandb|none
  NUM_WORKERS=6
  BATCH_SIZE=2
  ACCUM_BATCHES=4
  PRECISION=16-mixed
  CHECKPOINT_EVERY=5000
  VAL_EVERY=1000
  SA_SAVE_TOP_K=3
  SA_VAL_SAVE_AUDIO=1

Score-conditioning strategy:
  SA_UNFREEZE_PROFILE=hybrid|adaln|adapter|global|minimal
  SA_TRAINABLE_NAME_KEYS="continuous_score,to_scale_shift_gate,input_add_adapter"  # optional custom override
  SA_SELECTIVE_CFG_THRESHOLD=0.8  # sigma threshold for selective CFG (1.0 = always apply CFG)

Optimizer/scheduler:
  SA_LR=5e-5
  SA_WEIGHT_DECAY=1e-3
  SA_WARMUP_STEPS=1000
  SA_TOTAL_STEPS=300000
  SA_NUM_CYCLES=18
  SA_CFG_DROP_RATE=0.15

Validation generation:
  SA_VAL_NUM_SAMPLES=100
  SA_VAL_GEN_STEPS=50
  SA_VAL_CFG_SCALE=3.5

Example:
  SA_UNFREEZE_PROFILE=adaln RUN_NAME=sao_adaln_exp1 ./scripts/run_finetune_case3.sh
EOF
  exit 0
fi

cd "${REPO_ROOT}"

# -------- User-tunable defaults (can be overridden via env vars) --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-8,9}"
export MUSIC_RANKNET_ROOT="${MUSIC_RANKNET_ROOT:-/home/yonghyun/music-ranknet}"
export REWARD_MODEL_CKPT="${REWARD_MODEL_CKPT:-${MUSIC_RANKNET_ROOT}/checkpoints/ultimate_train_all(brainmusic).pt}"
export CLAP_MODEL_CKPT="${CLAP_MODEL_CKPT:-${MUSIC_RANKNET_ROOT}/checkpoints/music_audioset_epoch_15_esc_90.14.pt}"
export REWARD_THRESHOLDS_PATH="${REWARD_THRESHOLDS_PATH:-${MUSIC_RANKNET_ROOT}/data/processed/FMA_Scoring/reward_thresholds.json}"
export SA_WARMUP_STEPS="${SA_WARMUP_STEPS:-1000}"
export SA_TOTAL_STEPS="${SA_TOTAL_STEPS:-300000}"
export SA_NUM_CYCLES="${SA_NUM_CYCLES:-18}"
export SA_CFG_DROP_RATE="${SA_CFG_DROP_RATE:-0.15}"
export SA_UNFREEZE_PROFILE="${SA_UNFREEZE_PROFILE:-hybrid}"
export SA_LR="${SA_LR:-5e-5}"
export SA_WEIGHT_DECAY="${SA_WEIGHT_DECAY:-1e-3}"
export SA_VAL_NUM_SAMPLES="${SA_VAL_NUM_SAMPLES:-100}"
export SA_VAL_GEN_STEPS="${SA_VAL_GEN_STEPS:-50}"
export SA_VAL_CFG_SCALE="${SA_VAL_CFG_SCALE:-3.5}"
export SA_SAVE_TOP_K="${SA_SAVE_TOP_K:-3}"
export SA_VAL_SAVE_AUDIO="${SA_VAL_SAVE_AUDIO:-1}"
export SA_SELECTIVE_CFG_THRESHOLD="${SA_SELECTIVE_CFG_THRESHOLD:-0.8}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"

PYTHON_BIN="${PYTHON_BIN:-/home/yonghyun/miniconda3/envs/sao/bin/python3}"
RUN_NAME="${RUN_NAME:-sao_small_case3_${SA_UNFREEZE_PROFILE}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-./results/sao_small_case3}"
SAVE_DIR="${SAVE_DIR:-${SAVE_DIR_BASE}/${SA_UNFREEZE_PROFILE}}"
LOGGER_TYPE="${LOGGER_TYPE:-wandb}"
NUM_WORKERS="${NUM_WORKERS:-6}" # must be >= 1 due to persistent_workers=True

MODEL_CONFIG="${MODEL_CONFIG:-./checkpoints/sao_small/model_config_with_score.json}"
DATASET_CONFIG="${DATASET_CONFIG:-./configs/dataset_fma_scored.json}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-./configs/dataset_fma_scored.json}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:-./checkpoints/sao_small/model.safetensors}"

function require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    exit 1
  fi
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  echo "        Set PYTHON_BIN or activate your conda env first." >&2
  exit 1
fi

require_file "${MODEL_CONFIG}"
require_file "${DATASET_CONFIG}"
require_file "${PRETRAINED_CKPT_PATH}"
require_file "${REWARD_MODEL_CKPT}"
require_file "${CLAP_MODEL_CKPT}"
require_file "${REWARD_THRESHOLDS_PATH}"

if [[ "${LOGGER_TYPE}" == "wandb" ]] && command -v wandb >/dev/null 2>&1; then
  WANDB_STATUS="$(wandb status 2>/dev/null || true)"
  if [[ "${WANDB_STATUS}" == *'"api_key": null'* ]]; then
    echo "[WARN] wandb is not logged in. Run: wandb login" >&2
  fi
fi

if [[ "${NUM_WORKERS}" -lt 1 ]]; then
  echo "[ERROR] NUM_WORKERS must be >= 1 (current: ${NUM_WORKERS})" >&2
  exit 1
fi

echo "[INFO] Starting finetuning"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] SAVE_DIR=${SAVE_DIR}"
echo "[INFO] SA_UNFREEZE_PROFILE=${SA_UNFREEZE_PROFILE}"
echo "[INFO] SA_VAL_NUM_SAMPLES=${SA_VAL_NUM_SAMPLES}"
echo "[INFO] SA_VAL_GEN_STEPS=${SA_VAL_GEN_STEPS}"
echo "[INFO] SA_VAL_CFG_SCALE=${SA_VAL_CFG_SCALE}"
echo "[INFO] CHECKPOINT_EVERY=${CHECKPOINT_EVERY} SA_SAVE_TOP_K=${SA_SAVE_TOP_K} SA_VAL_SAVE_AUDIO=${SA_VAL_SAVE_AUDIO}"
echo "[INFO] SA_SELECTIVE_CFG_THRESHOLD=${SA_SELECTIVE_CFG_THRESHOLD}"
if [[ -n "${SA_TRAINABLE_NAME_KEYS:-}" ]]; then
  echo "[INFO] SA_TRAINABLE_NAME_KEYS=${SA_TRAINABLE_NAME_KEYS}"
fi

PYTHONPATH=. "${PYTHON_BIN}" ./finetune.py \
  --dataset-config "${DATASET_CONFIG}" \
  --val-dataset-config "${VAL_DATASET_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --pretrained-ckpt-path "${PRETRAINED_CKPT_PATH}" \
  --name "${RUN_NAME}" \
  --save-dir "${SAVE_DIR}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --accum-batches "${ACCUM_BATCHES:-4}" \
  --num-workers "${NUM_WORKERS}" \
  --precision "${PRECISION:-16-mixed}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --val-every "${VAL_EVERY:-1000}" \
  --logger "${LOGGER_TYPE}" \
  "$@"
