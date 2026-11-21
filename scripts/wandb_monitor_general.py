"""Run a command with W&B resource monitoring (Linux, single-node jobs).

Usage: python scripts/wandb_monitor_general.py --interval 30 -- bash run.sh
CPU/memory/I/O use the Slurm job's cgroup v2 when readable, otherwise the command's
process tree. GPU metrics describe allocated devices, not individual processes.
W&B working files live in a temporary directory and are removed after finish().
"""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def job_cgroup() -> Path | None:
    """Find the job ancestor, covering batch and other steps without crossing jobs."""
    try:
        relative = next(
            line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')
        )
        current = Path('/sys/fs/cgroup') / relative.lstrip('/')
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            for parent in (current, *current.parents):
                if re.fullmatch(rf'job[_-]?{re.escape(job_id)}(?:\.scope)?', parent.name):
                    if (parent / 'cpu.stat').is_file() and (parent / 'memory.current').is_file():
                        return parent
    except (OSError, StopIteration):
        pass
    return None


def cgroup_values(path: Path) -> dict[str, float]:
    values = {}
    cpu = dict(line.split() for line in (path / 'cpu.stat').read_text().splitlines())
    values['cpu_seconds'] = int(cpu['usage_usec']) / 1e6
    values['memory_bytes'] = int((path / 'memory.current').read_text())
    for filename, key in [('memory.peak', 'memory_peak_bytes'), ('memory.max', 'memory_limit_bytes')]:
        try:
            values[key] = int((path / filename).read_text())
        except (OSError, ValueError):
            pass
    try:
        # Kernel counters are per block device; these are block-layer totals.
        read_bytes = write_bytes = 0
        for line in (path / 'io.stat').read_text().splitlines():
            fields = dict(item.split('=') for item in line.split()[1:])
            read_bytes += int(fields.get('rbytes', 0))
            write_bytes += int(fields.get('wbytes', 0))
        values.update(read_bytes=read_bytes, write_bytes=write_bytes)
    except OSError:
        pass
    return values


class Resources:
    def __init__(self, pid: int, psutil):
        self.psutil = psutil
        self.pid = pid
        self.cgroup = job_cgroup()
        self.previous: dict[str, float] = {}
        self.last_time = time.monotonic()
        self.processes: dict[tuple[int, float], dict[str, float]] = {}
        self.cpu_count = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 1))
        self.memory_limit = float(os.environ.get('SLURM_MEM_PER_NODE', 0)) * 2**20
        if not self.memory_limit and os.environ.get('SLURM_MEM_PER_CPU'):
            self.memory_limit = float(os.environ['SLURM_MEM_PER_CPU']) * 2**20 * self.cpu_count
        self.scope = 'slurm_job_cgroup_v2' if self.cgroup else 'command_process_tree'

    def tree_values(self):
        """Retain last-seen counters of exited children to avoid counter resets."""
        try:
            root = self.psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
        except self.psutil.Error:
            processes = []
        memory = 0
        for process in processes:
            try:
                with process.oneshot():
                    times = process.cpu_times()
                    record = {'cpu_seconds': times.user + times.system}
                    memory += process.memory_info().rss
                    try:
                        io = process.io_counters()
                        record.update(read_bytes=io.read_bytes, write_bytes=io.write_bytes)
                    except (self.psutil.Error, AttributeError):
                        pass
                    self.processes[(process.pid, process.create_time())] = record
            except self.psutil.Error:
                continue
        totals = {'memory_bytes': memory}
        for record in self.processes.values():
            for key, value in record.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def sample(self):
        now = time.monotonic()
        elapsed = max(now - self.last_time, 1e-9)
        try:
            values = cgroup_values(self.cgroup) if self.cgroup else self.tree_values()
        except (OSError, ValueError, KeyError) as error:
            print(f'[monitor] resource sample skipped: {error}', file=sys.stderr)
            return {}
        metrics = {'job/memory_gib': values['memory_bytes'] / 2**30}
        limit = self.memory_limit or values.get('memory_limit_bytes', 0)
        if limit:
            metrics['job/memory_percent'] = 100 * values['memory_bytes'] / limit
        if 'memory_peak_bytes' in values:
            metrics['job/memory_peak_gib'] = values['memory_peak_bytes'] / 2**30
        if 'cpu_seconds' in self.previous and 'cpu_seconds' in values:
            cores = max(0, values['cpu_seconds'] - self.previous['cpu_seconds']) / elapsed
            metrics.update({'job/cpu_cores_used': cores, 'job/cpu_percent': 100 * cores / self.cpu_count})
        for key in ('read_bytes', 'write_bytes'):
            if key in values and key in self.previous:
                metrics[f'job/io_{key}_per_second'] = max(0, values[key] - self.previous[key]) / elapsed
        self.previous, self.last_time = values, now
        return metrics


def gpu_metrics() -> dict[str, float]:
    fields = 'index,uuid,utilization.gpu,memory.used,memory.total,power.draw'
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    # Slurm's physical device IDs take priority over CUDA's potentially remapped IDs.
    selection = os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')
    if selection is None:
        selection = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None if selection is None else set(selection.split(','))
    metrics = {}
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(',')]
        if len(parts) != 6:
            continue
        index, uuid, *raw = parts
        if allowed is not None and index not in allowed and not any(
            uuid.startswith(item) for item in allowed if item
        ):
            continue
        names = ('utilization_percent', 'vram_used_mib', 'vram_total_mib', 'power_watts')
        for name, value in zip(names, raw, strict=True):
            try:
                metrics[f'gpu/{uuid}/{name}'] = float(value)
            except ValueError:  # unsupported sensor, e.g. power on some devices
                pass
        used = metrics.get(f'gpu/{uuid}/vram_used_mib')
        total = metrics.get(f'gpu/{uuid}/vram_total_mib')
        if used is not None and total:
            metrics[f'gpu/{uuid}/vram_percent'] = 100 * used / total
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'geniac-baseline'))
    parser.add_argument('--name', default=os.environ.get('WANDB_NAME'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('provide a command after -- and an interval greater than zero')
    if os.environ.get('WANDB_MODE', 'online') != 'online':
        parser.error('this temporary-file monitor requires WANDB_MODE=online')
    try:
        import psutil
        import wandb
    except ImportError:
        parser.error('install monitoring dependencies: uv pip install -e ".[wandb]"')

    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    name = args.name or f"{os.environ.get('SLURM_JOB_NAME', 'job')}-{job_id}"
    received_signal = 0
    signal_time = None
    child = None

    def forward(signum, _frame):
        nonlocal received_signal, signal_time
        received_signal = signum
        if signal_time is None:
            signal_time = time.monotonic()
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)

    # No permanent monitoring CSV/JSON. Offline mode would lose unsynced data.
    with tempfile.TemporaryDirectory(prefix='geniac-wandb-') as directory:
        run = wandb.init(project=args.project, name=name, job_type='resource-monitor',
                         mode='online', dir=directory,
                         config={'slurm_job_id': job_id, 'interval_seconds': args.interval})
        exit_code = 1
        try:
            if received_signal:
                exit_code = 128 + received_signal
                return exit_code
            child = subprocess.Popen(command, start_new_session=True)
            if received_signal:
                forward(received_signal, None)
            resources = Resources(child.pid, psutil)
            run.config.update({'resource_scope': resources.scope, 'allocated_cpus': resources.cpu_count})
            if resources.cgroup is None:
                print('[monitor] cgroup unavailable; process-tree CPU/I/O may miss short-lived children; '
                      'RSS can double-count shared memory.', file=sys.stderr)
            started = time.monotonic()
            while True:
                metrics = {'job/elapsed_seconds': time.monotonic() - started,
                           **resources.sample(), **gpu_metrics()}
                try:
                    run.log(metrics)
                except Exception as error:
                    print(f'[monitor] W&B logging failed: {error}', file=sys.stderr)
                code = child.poll()
                if code is not None:
                    exit_code = code if code >= 0 else 128 - code
                    break
                deadline = time.monotonic() + args.interval
                while child.poll() is None and time.monotonic() < deadline:
                    if signal_time is not None and time.monotonic() - signal_time >= 10:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    try:
                        child.wait(timeout=min(0.5, max(0.001, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            run.summary['exit_code'] = exit_code
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    pass
            run.finish(exit_code=exit_code)
        return exit_code


if __name__ == '__main__':
    sys.exit(main())
