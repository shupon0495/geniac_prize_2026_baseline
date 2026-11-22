"""Resource counter interpretation, GPU filtering and wrapper lifecycle."""
import importlib.util
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'wandb_monitor', Path(__file__).resolve().parents[1] / 'scripts/wandb_monitor_general.py'
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def test_cgroup_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cpu.stat').write_text('usage_usec 1500000\nuser_usec 1000000\n')
            (root / 'memory.current').write_text('1024')
            (root / 'memory.max').write_text('4096')
            (root / 'io.stat').write_text('8:0 rbytes=100 wbytes=200 rios=1\n8:1 rbytes=20 wbytes=40\n')
            values = monitor.cgroup_values(root)
            self.assertEqual(values['cpu_seconds'], 1.5)
            self.assertEqual(values['memory_limit_bytes'], 4096)
            self.assertEqual(values['read_bytes'], 120)
            self.assertEqual(values['write_bytes'], 240)

    def test_allocated_cpu_normalization(self):
        with patch.object(monitor, 'job_cgroup', return_value=None), patch.dict(
            os.environ, {'SLURM_CPUS_ON_NODE': '16', 'SLURM_MEM_PER_NODE': '4'}
        ):
            resource = monitor.Resources(123, Mock())
        resource.previous = {'cpu_seconds': 10, 'read_bytes': 100}
        resource.last_time = 10
        with patch.object(monitor.time, 'monotonic', return_value=12), patch.object(
            resource, 'tree_values', return_value={'cpu_seconds': 26, 'memory_bytes': 2 * 2**20, 'read_bytes': 300}
        ):
            metrics = resource.sample()
        self.assertEqual(metrics['job/cpu_percent'], 50)
        self.assertEqual(metrics['job/memory_percent'], 50)
        self.assertEqual(metrics['job/io_read_bytes_per_second'], 100)

    def test_gpu_allocation_and_unsupported_sensor(self):
        result = SimpleNamespace(stdout='0, GPU-aaa, 90, 100, 200, 300\n1, GPU-bbb, 50, 20, 100, [N/A]\n')
        with patch.object(monitor.subprocess, 'run', return_value=result), patch.dict(
            os.environ, {'SLURM_JOB_GPUS': '1', 'SLURM_STEP_GPUS': ''}
        ):
            metrics = monitor.gpu_metrics()
        self.assertEqual(metrics['gpu/GPU-bbb/vram_percent'], 20)
        self.assertNotIn('gpu/GPU-bbb/power_watts', metrics)
        self.assertFalse(any('GPU-aaa' in key for key in metrics))

    def run_command(self, code):
        run = Mock()
        resources = Mock(scope='command_process_tree', cpu_count=1)
        resources.sample.return_value = {}
        run.summary = {}
        wandb = SimpleNamespace(init=Mock(return_value=run))
        command = [sys.executable, '-c', code]
        with patch.dict(sys.modules, {'wandb': wandb, 'psutil': Mock()}), patch.dict(
            os.environ, {'WANDB_MODE': 'online'}
        ), patch.object(sys, 'argv', ['monitor', '--interval', '0.01', '--', *command]), patch.object(
            monitor, 'Resources', return_value=resources
        ), patch.object(monitor, 'gpu_metrics', return_value={}):
            original = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
            try:
                status = monitor.main()
            finally:
                for sig, handler in original.items():
                    signal.signal(sig, handler)
        run.finish.assert_called_once_with(exit_code=status)
        self.assertFalse(Path(wandb.init.call_args.kwargs['dir']).exists())
        return status

    def test_command_exit_code_and_temporary_cleanup(self):
        self.assertEqual(self.run_command('raise SystemExit(7)'), 7)

    def test_child_signal_exit_code(self):
        self.assertEqual(self.run_command('import os, signal; os.kill(os.getpid(), signal.SIGTERM)'), 143)

    def test_wrapper_forwards_termination(self):
        self.assertEqual(self.run_command(
            'import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)'
        ), 143)


if __name__ == '__main__':
    unittest.main()
