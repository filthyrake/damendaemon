"""Real torchrun process-group deadline proof; deliberately no GPU inference."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import torch
from scripts.distributed_job import run_job


WORKER = '''import fcntl, json, os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
rank = int(os.environ['RANK'])
lock = open(os.path.join(sys.argv[1], f'rank{rank}.json'), 'w+')
fcntl.flock(lock, fcntl.LOCK_EX)
json.dump(dict(rank=rank, pid=os.getpid(), process_group=os.getpgrp()), lock)
lock.flush()
print(f'worker {rank} ready', flush=True)
while True: time.sleep(60)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = dict(passed=False, final_report=False, torch=str(torch.__version__),
                  scope='real CPU torchrun workers ignore TERM/INT; owned-group timeout cleanup; no GPU performance proof')
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-job-deadline-') as directory:
            worker = Path(directory)/'worker.py'
            worker.write_text(WORKER)
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                       '--run-path', str(worker), directory]
            started = time.monotonic()
            with patch('scripts.distributed_job.os.killpg', wraps=os.killpg) as signals:
                try:
                    run_job(command, timeout=20, termination_grace=1)
                except subprocess.TimeoutExpired as error:
                    report['stdout'] = error.stdout
                    report['cleanup_notes'] = getattr(error, '__notes__', [])
                    if report['cleanup_notes']:raise AssertionError('cleanup exceeded its deadline')
                else:raise AssertionError('hanging workers unexpectedly completed')
                report['signals'] = [dict(process_group=call.args[0], signal=int(call.args[1]))
                                     for call in signals.call_args_list]
            report['elapsed_seconds'] = time.monotonic()-started
            assert [event['signal'] for event in report['signals']] == [signal.SIGTERM, signal.SIGKILL]
            group = report['signals'][0]['process_group']
            assert group != os.getpgrp() and all(event['process_group'] == group for event in report['signals'])
            report['workers'] = []
            for rank in range(2):
                with (Path(directory)/f'rank{rank}.json').open() as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    row = json.load(lock)
                    assert row['rank'] == rank and row['process_group'] == group
                    report['workers'].append(row | dict(lock_released=True))
            report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['final_report'] = True
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':main()
