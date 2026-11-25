#!/usr/bin/env bash
#SBATCH --job-name=download-models
#SBATCH --partition=part-group_6d02c8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user_00003_6d02c8/shared-storage/members/Shun/logs/%x-%j.out
set -euo pipefail
cd /home/user_00003_6d02c8/shared-storage/members/Shun
geniac_prize_2026_baseline/.venv/bin/hf download Qwen/Qwen3.8-27B \
    --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir models/Qwen3.8-27B
geniac_prize_2026_baseline/.venv/bin/hf download allenai/Molmo2-8B --local-dir models/Molmo2-8B
