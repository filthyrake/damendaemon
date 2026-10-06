"""Run an owned HOMER experiment only with verified fan/thermal/memory guards.

Run /home/damen/fan-inference.sh on before using this launcher.
This does not replace a bounded systemd scope or prove universal driver safety.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from dllm.rocm_policy import host_available
from scripts.experiments.homer_cooling import CoolingGuard
from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if socket.gethostname().casefold() != 'homer':
        parser.error('this launcher is restricted to HOMER')
    if not command or not 1 <= args.timeout <= 1800 or args.report.exists():
        parser.error('provide a command, timeout1..1800 and a new report path')
    report = dict(command=command, timeout_seconds=args.timeout, cooling={}, watchdog={},
                  phase='preflight', cleanup_complete=False)
    child = watchdog = None
    started = time.monotonic()

    def save():
        temporary = args.report.with_suffix(args.report.suffix+'.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.report)

    def stop():
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def interrupted(signum, frame):
        stop()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        cooling = CoolingGuard(report['cooling'])
        cooling.check(startup=True)
        report['host_available_before'] = host_available()
        if report['host_available_before'] < 80*1024**3:
            raise MemoryError('requires80GiB initial host headroom')
        save()
        with args.report.with_suffix('.stdout').open('x') as stdout, args.report.with_suffix('.stderr').open('x') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            report.update(phase='running', child_pid=child.pid)
            watchdog = HostMemoryWatchdog(stop, report['watchdog'], read=cooling.read_host_available).start()
            while child.poll() is None:
                report['seconds'] = time.monotonic()-started
                if report['seconds'] >= args.timeout:
                    report['timed_out'] = True
                    stop()
                save()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            report['exit_code'] = child.returncode
            report['phase'] = 'completed' if child.returncode == 0 and not report['watchdog']['triggered'] else 'failed'
    except BaseException as error:
        report.update(phase='failed', error=repr(error))
        raise
    finally:
        stop()
        if child is not None:
            child.wait()
        if watchdog is not None:
            watchdog.close()
        report.update(cleanup_complete=child is None or child.poll() is not None,
                      seconds=time.monotonic()-started, host_available_after=host_available())
        save()
    if report['phase'] != 'completed':
        raise SystemExit('experiment failed; retained guard report')


if __name__ == '__main__':
    main()
