#!/usr/bin/env bash
#SBATCH --job-name=collect
#SBATCH --partition=part-group_6d02c8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user_00003_6d02c8/shared-storage/members/Shun/logs/%x-%j.out
set -euo pipefail
cd /home/user_00003_6d02c8/shared-storage/members/Shun/geniac_prize_2026_baseline
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
export VLLM_USE_FLASHINFER_SAMPLER=0

# 読むファイルを確認
SPEC="${1:?Usage: sbatch jobs/collect_job.sh <spec.yaml>}"
if [[ ! -f "$SPEC" ]]; then
    echo "設定ファイルが見つかりません: $SPEC" >&2
    exit 1
fi

COLLECT_NAME="$(basename "$SPEC" .yaml)"
RUN_DIR="runs/collect/${COLLECT_NAME}-${SLURM_JOB_ID}"

.venv/bin/geniac-baseline collect plan \
    --spec "$SPEC" \
    --output-dir "$RUN_DIR"

bash jobs/with_wandb.sh bash "$RUN_DIR/run.sh"
