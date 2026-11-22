#!/usr/bin/env bash
# Put this file at: shared-storage/members/<user>/start_web_ui.sh
# Run on the login node: bash start_web_ui.sh
# Requirements: sbatch/srun, uv, Node.js/npm, and installed web-ui dependencies.

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FRONT_PORT=5173
BACK_PORT=8200

# The same script is submitted by sbatch and re-executed on the compute node.
if [[ "${1:-}" != "--worker" ]]; then
    if (( $# != 0 )); then
        echo "Usage: bash $0" >&2
        exit 2
    fi

    # Default layout: members/<user>/geniac_prize_2026_baseline/geniac_prize_2026_code/
    # Optional override: CAPX_PROJECT_DIR=/absolute/path/to/geniac_prize_2026_code
    if [[ -n "${CAPX_PROJECT_DIR:-}" ]]; then
        CANDIDATES=("$CAPX_PROJECT_DIR")
    else
        CANDIDATES=(
            "$SCRIPT_DIR/geniac_prize_2026_baseline/geniac_prize_2026_code"
            "$SCRIPT_DIR/geniac_prize_2026_code"
            "$SCRIPT_DIR"
        )
    fi

    PROJECT_DIR=""
    for candidate in "${CANDIDATES[@]}"; do
        if [[ -f "$candidate/web-ui/package.json" && -f "$candidate/capx/web/server.py" ]]; then
            PROJECT_DIR="$(cd -- "$candidate" && pwd -P)"
            break
        fi
    done
    if [[ -z "$PROJECT_DIR" ]]; then
        echo "ERROR: Could not find geniac_prize_2026_code/." >&2
        echo "Expected: $SCRIPT_DIR/geniac_prize_2026_baseline/geniac_prize_2026_code" >&2
        echo "Or set: CAPX_PROJECT_DIR=/path/to/geniac_prize_2026_code" >&2
        exit 1
    fi

    # Keep all logs out of the cloned Git repository, under each member's directory.
    LOG_BASE="$SCRIPT_DIR/logs/web-ui"
    mkdir -p -- "$LOG_BASE"
    chmod 700 "$SCRIPT_DIR/logs" "$LOG_BASE"

    export CAPX_PROJECT_DIR="$PROJECT_DIR" CAPX_WEB_UI_LOG_DIR="$LOG_BASE"
    export FRONT_PORT BACK_PORT

    SBATCH_ARGS=(
        --parsable
        --job-name=capx-web-ui
        --nodes=1
        --ntasks=1
        --cpus-per-task="${CPUS:-4}"
        --gpus="${GPUS:-1}"
        --mem="${MEM:-8G}"
        --time="${TIME_LIMIT:-02:00:00}"
        --chdir="$PROJECT_DIR"
        --output="$LOG_BASE/slurm-%j.out"
        --export=ALL
    )
    if [[ -n "${PARTITION:-}" ]]; then
        SBATCH_ARGS+=(--partition="$PARTITION")
    fi

    JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" --worker)"
    JOB_ID="${JOB_ID%%;*}"  # Some Slurm installations return jobid;cluster.
    echo "Submitted Slurm job: $JOB_ID"
    echo "Code: $PROJECT_DIR"
    echo "Node: squeue -j $JOB_ID -o '%i %T %N'"
    echo "Logs: $LOG_BASE/$JOB_ID/{backend,frontend}.log"
    echo "Slurm: $LOG_BASE/slurm-$JOB_ID.out"
    exit 0
fi

# From here onward, sbatch runs this script on one allocated compute node.
: "${SLURM_JOB_ID:?This mode must be run by sbatch}"
: "${CAPX_PROJECT_DIR:?Missing project path from submission}"
: "${CAPX_WEB_UI_LOG_DIR:?Missing log path from submission}"

cd -- "$CAPX_PROJECT_DIR"
JOB_LOG_DIR="$CAPX_WEB_UI_LOG_DIR/$SLURM_JOB_ID"
mkdir -p -- "$JOB_LOG_DIR"
chmod 700 "$JOB_LOG_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "Backend: 127.0.0.1:$BACK_PORT"
echo "Frontend: 127.0.0.1:$FRONT_PORT"
echo "Logs: $JOB_LOG_DIR"

# Separate steps, but both use the SAME Slurm allocation and compute node.
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
    bash -c '
        set -e
        cd "$CAPX_PROJECT_DIR"
        export PATH="$HOME/.local/bin:$PATH"
        exec uv run python -u -m capx.web.server --host 127.0.0.1 --port "$BACK_PORT"
    ' > "$JOB_LOG_DIR/backend.log" 2>&1 &
BACK_SRUN_PID=$!

srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
    bash -c '
        set -e
        export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
        if [[ -s "$NVM_DIR/nvm.sh" ]]; then
            source "$NVM_DIR/nvm.sh"
        fi
        cd "$CAPX_PROJECT_DIR/web-ui"
        exec npm run dev -- --host 127.0.0.1 --port "$FRONT_PORT" --strictPort
    ' > "$JOB_LOG_DIR/frontend.log" 2>&1 &
FRONT_SRUN_PID=$!

cleanup() {
    trap - EXIT
    kill "$BACK_SRUN_PID" "$FRONT_SRUN_PID" 2>/dev/null || true
    wait "$BACK_SRUN_PID" "$FRONT_SRUN_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Do not leave a half-running web UI occupying allocated resources.
set +e
wait -n "$BACK_SRUN_PID" "$FRONT_SRUN_PID"
EXIT_CODE=$?
set -e
echo "A service stopped (exit code $EXIT_CODE); stopping the other service."
if [[ "$EXIT_CODE" -eq 0 ]]; then EXIT_CODE=1; fi
exit "$EXIT_CODE"
