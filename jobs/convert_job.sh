#!/usr/bin/env bash
#SBATCH --job-name=convert
#SBATCH --partition=part-group_6d02c8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user_00003_6d02c8/shared-storage/members/Shun/logs/%x-%j.out
set -euxo pipefail
cd /home/user_00003_6d02c8/shared-storage/members/Shun/geniac_prize_2026_baseline
GB=.venv/bin/geniac-baseline
SRC=runs/collect_libero/sources.yaml
TOK=/home/user_00003_6d02c8/shared-storage/members/Shun/models/Qwen3.8-27B  
"$GB" convert rows --sources $SRC --out runs/data/rows
"$GB" convert rows --sources $SRC --include failure --out runs/data/fail_rows
"$GB" convert tag --src runs/data/rows --out runs/data/tagged --tokenizer $TOK
"$GB" convert tag --src runs/data/fail_rows --out runs/data/fail_tagged --tokenizer $TOK
"$GB" convert split --src runs/data/tagged --out runs/data/split
"$GB" convert validate runs/data/split/sft.jsonl runs/data/split/val.jsonl
"$GB" pairs --success runs/data/split --failure runs/data/fail_tagged --out runs/data/pairs