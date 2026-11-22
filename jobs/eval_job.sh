#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --partition=part-group_6d02c8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user_00003_6d02c8/shared-storage/members/Shun/logs/%x-%j.out
set -euo pipefail
cd /home/user_00003_6d02c8/shared-storage/members/Shun/geniac_prize_2026_baseline

SPEC="${1:?Usage: sbatch jobs/eval_job.sh <spec.yaml>}"
if [[ ! -f "$SPEC" ]]; then
    echo "設定ファイルが見つかりません: $SPEC" >&2
    exit 1
fi

EVAL_NAME="$(basename "$SPEC" .yaml)"
RUN_DIR="runs/eval/${EVAL_NAME}_${SLURM_JOB_ID}"

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
export VLLM_USE_FLASHINFER_SAMPLER=0

.venv/bin/geniac-baseline eval plan \
    --spec "$SPEC" \
    --output-dir "$RUN_DIR"

bash jobs/with_wandb.sh bash "$RUN_DIR/run.sh"

.venv/bin/geniac-baseline eval score \
    "$RUN_DIR" \
    --json "$RUN_DIR/score.json"