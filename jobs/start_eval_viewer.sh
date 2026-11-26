#!/usr/bin/env bash
# Submit from a login node; worker runs one CPU-only Gradio process on its allocated node.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
self="$repo/jobs/start_eval_viewer.sh"
worker=0
prepare=0
reuse=0
results="${VIEWER_ROOT:-}"
state="${VIEWER_STATE_DIR:-}"
partition="${VIEWER_PARTITION:-}"
cpus="${VIEWER_CPUS:-2}"
memory="${VIEWER_MEMORY:-8G}"
limit="${VIEWER_TIME:-04:00:00}"
port="${VIEWER_PORT:-7860}"
gpus="${VIEWER_GPUS:-0}"
python="${VIEWER_PYTHON:-python3}"
usage() {
  cat <<'EOF'
bash jobs/start_eval_viewer.sh --root RESULTS --state-dir OWN_STATE [options]
  --partition NAME   CPU partition (omitted: cluster default)
  --cpus N --memory SIZE --time LIMIT --port N --gpus N
  --python PYTHON     Bootstrap interpreter / uv Python version (default python3)
  --prepare-only      Prepare dedicated venv without submitting or starting the app
  --reuse-env         Use prepared venv, do not access dependency registries
Environment: VIEWER_ROOT, VIEWER_STATE_DIR, VIEWER_PARTITION, VIEWER_CPUS,
VIEWER_MEMORY, VIEWER_TIME, VIEWER_PORT, VIEWER_GPUS, VIEWER_PYTHON.
EOF
}
while (($#)); do
  case "$1" in
    --worker) worker=1; shift ;;
    --prepare-only) prepare=1; shift ;;
    --reuse-env) reuse=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --root|--state-dir|--partition|--cpus|--memory|--time|--port|--gpus|--python)
      (($# >= 2)) || { echo "missing value: $1" >&2; exit 2; }
      case "$1" in
        --root) results="$2" ;; --state-dir) state="$2" ;; --partition) partition="$2" ;;
        --cpus) cpus="$2" ;; --memory) memory="$2" ;; --time) limit="$2" ;;
        --port) port="$2" ;; --gpus) gpus="$2" ;; --python) python="$2" ;;
      esac
      shift 2 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$results" && -n "$state" ]] || { usage >&2; exit 2; }
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$gpus" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || exit 2
(( port >= 1 && port <= 65535 )) || exit 2
# Validate before ANY mkdir, uv cache, venv or log write. Canonical paths also catch symlink aliases.
# The bootstrap interpreter is only needed for this stdlib-only boundary check.
bootstrap="${VIEWER_BOOTSTRAP_PYTHON:-python3}"
mapfile -t paths < <("$bootstrap" - "$results" "$state" "$repo" <<'PY'
import pathlib, sys
root, state, repo = [pathlib.Path(p).expanduser().resolve() for p in sys.argv[1:]]
if not root.is_dir() or root == state or root in state.parents or state in root.parents:
    sys.exit("error: results root and state directory must be separate, non-overlapping directories")
if root == repo or root in repo.parents:
    sys.exit("error: viewer checkout must be outside the results root")
for sub in ("venv", "logs", "tmp", "cache", "uv-cache", "gradio", "huggingface", "runtime"):
    target = state / sub
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        sys.exit(f"error: state subdirectory must be a normal directory: {target}")
print(root)
print(state)
PY
)
[[ ${#paths[@]} -eq 2 ]] || exit 2
results="${paths[0]}"
state="${paths[1]}"
umask 077
mkdir -p "$state/logs" "$state/tmp" "$state/cache" "$state/uv-cache" "$state/gradio" "$state/huggingface" "$state/runtime"
export UV_CACHE_DIR="$state/uv-cache" TMPDIR="$state/tmp" XDG_CACHE_HOME="$state/cache"
export GRADIO_TEMP_DIR="$state/gradio" GRADIO_ANALYTICS_ENABLED=False HF_HOME="$state/huggingface"
export PYTHONDONTWRITEBYTECODE=1
cd "$state/runtime"

child=""
cleanup() {
  if [[ -n "$child" ]]; then
    # Each long-lived command has its own session, including uv/build subprocesses.
    kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 -- "-$child" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
run_child() {
  command -v setsid >/dev/null || { echo "error: setsid (util-linux) not found" >&2; exit 1; }
  setsid "$@" &
  child=$!
  wait "$child"
  child=""
}
setup_env() {
  command -v uv >/dev/null || { echo "error: uv not found" >&2; exit 1; }
  if (( reuse )); then
    [[ -x "$state/venv/bin/geniac-eval-viewer" ]] || { echo "error: prepare viewer venv first" >&2; exit 1; }
  else
    if [[ ! -x "$state/venv/bin/python" ]]; then
      run_child uv venv --python "$python" "$state/venv"
    fi
    run_child uv pip install --python "$state/venv/bin/python" "$repo[viewer]"
  fi
}
if (( prepare )); then
  setup_env
  echo "Viewer venv prepared: $state/venv; submit with --reuse-env"
  exit 0
fi
if (( ! worker )); then
  command -v sbatch >/dev/null || { echo "error: sbatch not found; run on the login node" >&2; exit 1; }
  opts=(--parsable --job-name=eval-viewer --nodes=1 --ntasks=1 "--cpus-per-task=$cpus"
    "--mem=$memory" "--time=$limit" --no-requeue "--chdir=$state/runtime" "--output=$state/logs/%x-%j.out")
  [[ -z "$partition" ]] || opts+=("--partition=$partition")
  (( gpus == 0 )) || opts+=("--gpus=$gpus")
  args=(--worker --root "$results" --state-dir "$state" --port "$port" --python "$python")
  (( ! reuse )) || args+=(--reuse-env)
  job="$(sbatch "${opts[@]}" "$self" "${args[@]}")"
  job="${job%%;*}"
  echo "ジョブID: $job  ポート: $port  ログ: $state/logs/eval-viewer-$job.out"
  echo "ノード確認: squeue -h -j $job -o %N"
  echo "停止: scancel $job"
  echo "接続情報とSSH転送例はworkerログに出力します。"
  exit 0
fi
echo "ジョブID: ${SLURM_JOB_ID:-local}  ノード: $(hostname)  ポート: $port"
echo "ログ: $state/logs/eval-viewer-${SLURM_JOB_ID:-local}.out"
echo "停止: scancel ${SLURM_JOB_ID:-JOB_ID}"
echo "手元PCで、接続経路に合わせて実行する例:"
echo "ssh -N -J <login-alias> -L 127.0.0.1:$port:127.0.0.1:$port <compute-alias>"
echo "計算ノード自身のlocalhostへ到達するSSH経路が必要です。"
setup_env
run_child "$state/venv/bin/geniac-eval-viewer" --root "$results" --state-dir "$state" --port "$port"
