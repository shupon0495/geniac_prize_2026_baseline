from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "jobs" / "start_eval_viewer.sh"


@pytest.mark.parametrize("gpu", [0, 2])
def test_sbatch_stub_spaces_and_gpu_default(tmp_path, gpu):
    root = tmp_path / "shared results"
    state = tmp_path / "own state"
    root.mkdir()
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "sbatch"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['STUB_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\nprint('123;cluster')\n"
    )
    stub.chmod(0o755)
    record = tmp_path / "args.json"
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "STUB_ARGS": str(record),
        "VIEWER_BOOTSTRAP_PYTHON": sys.executable,
    }
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--partition",
            "cpu-test",
            "--gpus",
            str(gpu),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(record.read_text())
    assert "--nodes=1" in args and "--ntasks=1" in args
    assert "--partition=cpu-test" in args
    assert ("--gpus=2" in args) == bool(gpu)
    assert not any("gpu" in a for a in args) if not gpu else True
    assert args[args.index("--root") + 1] == str(root)
    assert args[args.index("--state-dir") + 1] == str(state)
    assert "scancel 123" in result.stdout
    assert list(root.iterdir()) == []


def test_script_rejects_state_inside_results_without_creating_it(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--root", str(root), "--state-dir", str(root / "state")],
        env={**os.environ, "VIEWER_BOOTSTRAP_PYTHON": sys.executable},
        capture_output=True,
    )
    assert result.returncode != 0
    assert list(root.iterdir()) == []
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_worker_term_stops_child(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    state = tmp_path / "state"
    executable = state / "venv" / "bin" / "geniac-eval-viewer"
    executable.parent.mkdir(parents=True)
    marker = state / "child.pid"
    stopped = state / "stopped"
    executable.write_text(
        f"#!{sys.executable}\nimport os, signal, sys, time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        f"def stop(*_):\n    Path({str(stopped)!r}).touch()\n    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\ntime.sleep(60)\n"
    )
    executable.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(SCRIPT), "--worker", "--reuse-env", "--root", str(root), "--state-dir", str(state)],
        env={**os.environ, "VIEWER_BOOTSTRAP_PYTHON": sys.executable},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            if process.poll() is not None:
                pytest.fail(process.communicate()[1].decode())
            time.sleep(0.05)
        assert marker.exists()
        process.terminate()
        process.wait(timeout=10)
        assert process.returncode != 0 and stopped.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        assert list(root.iterdir()) == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
