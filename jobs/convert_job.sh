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
COLLECT_DIR="${1:?Usage: sbatch jobs/convert_job.sh <collect_dir>}"
COLLECT_DIR="$(realpath -- "$COLLECT_DIR")"
SRC="$COLLECT_DIR/sources.yaml"
if [[ ! -f "$SRC" ]]; then
    echo "収集元ファイルが見つかりません: $SRC" >&2
    exit 1
fi

COLLECT_NAME="$(basename -- "$COLLECT_DIR")"
OUT_DIR="runs/convert/${COLLECT_NAME}-convert-${SLURM_JOB_ID}"
mkdir -p runs/convert
mkdir -- "$OUT_DIR"
printf '%s\n' "$COLLECT_DIR" > "$OUT_DIR/collect_source.txt"

TOK=/home/user_00003_6d02c8/shared-storage/members/Shun/models/Qwen3.8-27B
"$GB" convert rows --sources "$SRC" --out "$OUT_DIR/rows"
"$GB" convert rows --sources "$SRC" --include failure --out "$OUT_DIR/fail_rows"
"$GB" convert tag --src "$OUT_DIR/rows" --out "$OUT_DIR/tagged" --tokenizer "$TOK"
"$GB" convert tag --src "$OUT_DIR/fail_rows" --out "$OUT_DIR/fail_tagged" --tokenizer "$TOK"
"$GB" convert split --src "$OUT_DIR/tagged" --out "$OUT_DIR/split"
"$GB" convert validate "$OUT_DIR/split/sft.jsonl" "$OUT_DIR/split/val.jsonl"
"$GB" pairs --success "$OUT_DIR/split" --failure "$OUT_DIR/fail_tagged" --out "$OUT_DIR/pairs"
