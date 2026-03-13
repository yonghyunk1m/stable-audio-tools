#!/usr/bin/env bash
# Prune checkpoints to free disk space. Keeps only the last N checkpoints by step.
#
# Usage:
#   ./scripts/cleanup_checkpoints.sh [CHECKPOINT_DIR] [KEEP_N]
#
# Examples:
#   ./scripts/cleanup_checkpoints.sh
#     Uses default: results/sao_small_case3/music-steerability-study/sao_small_case3_0301_1923/checkpoints
#     Keeps last 1 checkpoint
#
#   ./scripts/cleanup_checkpoints.sh /path/to/checkpoints 3
#     Keeps last 3 checkpoints in given dir
#
#   DRY_RUN=1 ./scripts/cleanup_checkpoints.sh
#     Show what would be deleted without actually deleting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT_DIR="${1:-${REPO_ROOT}/results/sao_small_case3/music-steerability-study/sao_small_case3_0301_1923/checkpoints}"
KEEP_N="${2:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

# List .ckpt files, sort by version (epoch/step), keep last KEEP_N
# sort -V handles epoch=0-step=1000, epoch=27-step=181000 etc.
mapfile -t all_ckpts < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V)

total=${#all_ckpts[@]}
if [[ total -eq 0 ]]; then
  echo "[INFO] No checkpoints found in ${CHECKPOINT_DIR}"
  exit 0
fi

# to_keep = last KEEP_N
# to_delete = the rest
keep_count=$(( total < KEEP_N ? total : KEEP_N ))
delete_count=$(( total - keep_count ))

to_keep=()
for (( i = total - keep_count; i < total; i++ )); do
  to_keep+=("${all_ckpts[$i]}")
done

to_delete=()
for (( i = 0; i < total - keep_count; i++ )); do
  to_delete+=("${all_ckpts[$i]}")
done

echo "[INFO] Checkpoint dir: ${CHECKPOINT_DIR}"
echo "[INFO] Total checkpoints: ${total}"
echo "[INFO] Keeping ${keep_count}:"
for k in "${to_keep[@]}"; do
  echo "  - $(basename "$k")"
done
echo "[INFO] Deleting ${delete_count} checkpoints"

if [[ delete_count -eq 0 ]]; then
  echo "[INFO] Nothing to delete."
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] Would delete:"
  for f in "${to_delete[@]}"; do
    echo "  - $f"
  done
  echo "[DRY RUN] Run without DRY_RUN=1 to actually delete."
  exit 0
fi

echo "[WARN] About to delete ${delete_count} files. Press Ctrl+C within 5 seconds to cancel..."
sleep 5

freed=0
for f in "${to_delete[@]}"; do
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  rm -f "$f"
  freed=$((freed + size))
  echo "  Deleted: $(basename "$f")"
done

freed_gb=$((freed / 1024 / 1024 / 1024))
echo "[DONE] Freed approximately ${freed_gb} GB"
