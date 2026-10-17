"""Admit short CUDA checks with incumbent health and resource reserves.

Polling is containment, not resource isolation or a latency guarantee. Commands
must remain in their owned process group; do not launch daemonizing children.
Use a separate workload allocator cap as well as this external reserve monitor.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

from damendaemon.rocm_policy import host_available


def read_json(url):
    with urlopen(url, timeout=1) as response:
        data = response.read(65537)
    if len(data) > 65536:
        raise ValueError('oversized service telemetry')
    return json.loads(data)


def check_service(health, slots, allow_busy=False):
    if not isinstance(health, dict) or health.get('status') != 'ok':
        raise RuntimeError('incumbent service unhealthy')
    if not isinstance(slots, list) or not slots:
        raise RuntimeError('missing incumbent slots')
    for slot in slots:
        if (not isinstance(slot, dict) or type(slot.get('is_processing')) is not bool
                or (slot['is_processing'] and not allow_busy)):
            raise RuntimeError('incumbent slot busy or invalid')


def process_identity(pid):
    # comm may contain spaces and parentheses; starttime follows the last ')'.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields[19]


class Probe:
    def __init__(self, uuids, service=None, incumbent=None, allow_busy=False):
        if isinstance(uuids, str):
            uuids = [uuids]
        self.uuids, self.service, self.incumbent = uuids, service, incumbent
        self.identity = process_identity(incumbent) if incumbent is not None else None
        self.allow_busy = allow_busy

    def __call__(self, startup=False):
        if self.incumbent is not None:
            if process_identity(self.incumbent) != self.identity:
                raise RuntimeError('incumbent process changed')
            service = self.service
            check_service(read_json(service+'/health'), read_json(service+'/slots'), self.allow_busy)
        devices = []
        for uuid in self.uuids:
            output = subprocess.run([
                'nvidia-smi', '--id='+uuid,
                '--query-gpu=uuid,temperature.gpu,memory.free',
                '--format=csv,noheader,nounits'], check=True, capture_output=True,
                text=True, timeout=2).stdout
            identity, temperature, free = [part.strip() for part in output.strip().split(',')]
            temperature, free = int(temperature), int(free)
            if identity != uuid or not 0 <= temperature < 65:
                raise RuntimeError('GPU identity or temperature gate failed')
            if free < (4096 if startup else 2560):
                raise MemoryError('GPU reserve gate failed')
            devices.append(dict(uuid=uuid, temperature_c=temperature, gpu_free_mib=free))
        available = host_available()
        if available < 16*1024**3:
            raise MemoryError('host reserve gate failed')
        return dict(devices=devices,
                    temperature_c=max(device['temperature_c'] for device in devices),
                    gpu_free_mib=min(device['gpu_free_mib'] for device in devices),
                    host_available_bytes=available)


def run(command, probe, timeout, report, env=None):
    child = None
    started = time.monotonic()
    report.update(phase='preflight', child_started=False, cleanup_complete=False)

    def stop():
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        report['before'] = probe(startup=True)
        child = subprocess.Popen(command, env=env, start_new_session=True)
        report.update(phase='running', child_started=True, child_pid=child.pid)
        while child.poll() is None:
            if time.monotonic()-started >= timeout:
                raise TimeoutError('experiment deadline')
            report['latest'] = probe()
            try:
                child.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
        report['exit_code'] = child.returncode
        report['after'] = probe()
        if child.returncode:
            raise RuntimeError('owned command failed')
        report['phase'] = 'completed'
    except Exception as error:
        report.update(phase='failed', error=str(error))
    finally:
        # Also remove descendants that survived a successfully exited leader.
        stop()
        if child is not None:
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(cleanup_complete=True, seconds=time.monotonic()-started)
    return report['phase'] == 'completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-busy', action='store_true',
                        help='allow healthy active slots; may affect incumbent latency')
    parser.add_argument('--gpu', action='append', required=True,
                        help='explicit GPU UUID; repeat once for a two-device run')
    parser.add_argument('--service', default=None,
                        help='incumbent health/slots endpoint; omit when no incumbent runs')
    parser.add_argument('--incumbent-pid', type=int, default=None,
                        help='incumbent PID to protect; omit when no incumbent runs')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 1 <= args.timeout <= 1800:
        parser.error('command and timeout 1..1800 required')
    if args.service is None and args.incumbent_pid is not None:
        parser.error('--service is required with --incumbent-pid')
    if args.service is not None and args.incumbent_pid is None:
        parser.error('--incumbent-pid is required with --service')
    if args.incumbent_pid is not None and args.incumbent_pid <= 1:
        parser.error('positive incumbent PID required')
    if (len(args.gpu) not in (1, 2) or len(set(args.gpu)) != len(args.gpu)
            or any(not uuid.startswith('GPU-') or ',' in uuid for uuid in args.gpu)):
        parser.error('one or two distinct explicit GPU UUIDs required')
    # Reserve the evidence file before admitting a child; never overwrite a run.
    with args.report.open('x') as output:
        report = dict(gpu_uuids=args.gpu, incumbent_pid=args.incumbent_pid,
                      incumbent_service=args.service,
                      timeout_seconds=args.timeout, isolation=False, allow_busy=args.allow_busy)
        env = {k: v for k, v in os.environ.items() if not k.startswith('DAMENDAEMON_')}
        env.update(CUDA_VISIBLE_DEVICES=','.join(args.gpu), DAMENDAEMON_DEVICE='cuda',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', DAMENDAEMON_SHARED_CUDA_GUARD='1')
        try:
            passed = run(command, Probe(args.gpu, args.service, args.incumbent_pid, args.allow_busy),
                         args.timeout, report, env)
        except Exception as error:
            passed = False
            report.update(phase='failed', child_started=False, error=str(error))
        json.dump(report, output, indent=2)
        output.write('\n')
    if not passed:
        raise SystemExit('shared CUDA gate refused or stopped run; see report')


if __name__ == '__main__':
    main()
