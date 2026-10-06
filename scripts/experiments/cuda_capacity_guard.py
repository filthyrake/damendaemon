"""Read-only A30 preflight/thermal telemetry for the isolated Q4 capacity gate."""
import os
import subprocess
import time

from dllm.rocm_policy import host_available


class A30CapacityGuard:
    def __init__(self, report):
        self.report = report
        self.uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not self.uuid.startswith('GPU-') or ',' in self.uuid:
            raise ValueError('A30 capacity guard requires exactly one explicit GPU UUID')
        self.next_sample = 0
        report.update(gpu_uuid=self.uuid, cutoff_celsius=85, samples=0,
                      maximum_observed_celsius=None, maximum_device_used_bytes=None)

    def read(self, startup=False):
        available = host_available()
        now = time.monotonic()
        if startup or now >= self.next_sample:
            result = subprocess.run(['nvidia-smi', '--id='+self.uuid,
                '--query-gpu=name,temperature.gpu,memory.used,memory.free',
                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                check=True, timeout=3)
            name, temperature, used, free = [part.strip() for part in result.stdout.strip().split(',')]
            temperature, used, free = int(temperature), int(used)*1024**2, int(free)*1024**2
            if name != 'NVIDIA A30' or not 0 <= temperature <= 120 or min(used, free) < 0:
                raise RuntimeError('invalid A30 telemetry')
            self.report.update(samples=self.report['samples']+1, celsius=temperature,
                               device_used_bytes=used, device_free_bytes=free)
            self.report['maximum_observed_celsius'] = max(
                temperature, self.report['maximum_observed_celsius'] or 0)
            self.report['maximum_device_used_bytes'] = max(
                used, self.report['maximum_device_used_bytes'] or 0)
            self.next_sample = now+1
            if temperature >= (70 if startup else 85):
                raise RuntimeError('A30 capacity thermal cutoff')
            if startup and free < 22*1024**3:
                raise MemoryError('A30 capacity preflight requires22GiB free GPU memory')
        if startup and available < 100*1024**3:
            raise MemoryError('A30 capacity preflight requires100GiB available host memory')
        return available
