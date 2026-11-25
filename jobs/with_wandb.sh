#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$root/.venv/bin/python" \
    "$root/scripts/wandb_monitor_general.py" \
    --interval 30 -- "$@"