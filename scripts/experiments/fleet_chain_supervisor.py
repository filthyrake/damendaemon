"""#732: bounded fleet-chain supervision — stage-crash recovery from parked dumps.

Runs ON the stage host, detached. Owns every guarded attempt as a direct
child (``Popen`` handles — death detection never uses a pgrep pattern),
polls the guard report's ``phase`` field as the authoritative completion
signal, and on a failed/killed attempt relaunches ONLY the affected
sub-cell from its parked inbound handoff file with fresh report paths,
up to ``--attempts``. The upstream stage is never rerun; the parked
inbound file is only read (md5-verified before and after by the caller).

Pre-declared gates (#732):
1. class-A (mid-compute SIGKILL): supervisor relaunches; final worker
   report passed; ``sent_hash`` == ``--expected-hash`` (the uninterrupted
   control's recorded hash); final guard clean.
2. class-B (kill during startup, before the first layer event): same
   recovery from the intact inbound file; hash matches.
3. Detection is report/process-handle based (structural: no pgrep /
   liveness-probe code paths exist in this module).
4. Every attempt's guard + worker reports retained; fresh paths per
   attempt; an existing output path refuses the launch up front.

No throughput/SLO claim — recovery-correctness validation only.
"""
if not __debug__:
    raise RuntimeError('fleet_chain_supervisor asserts report invariants; '
                       'run without -O')

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

WORKER_MODULE = 'scripts.experiments.glm_q4_fleet_worker'
GUARD_MODULE = 'scripts.experiments.guard_shared_cuda'
DONE_PHASES = ('completed', 'failed', 'timeout', 'stopped')


def _atomic_write(path, payload):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=1)
    os.replace(tmp, path)


def _load_guard_phase(path):
    """Guard reports can carry trailing text after the JSON object."""
    try:
        text = open(path).read()
    except OSError:
        return None
    start = text.find('{')
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None
    return obj.get('phase')


def _worker_verdict(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True,
                        help='path prefix: {label}-out-{n}.json etc. (must be unused)')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--gpu', required=True, help='guard GPU UUID')
    parser.add_argument('--guard-timeout', type=float, default=1800.)
    parser.add_argument('--thermal-limit-c', type=float, default=90)
    parser.add_argument('--gpu-reserve-mib', type=int, default=0)
    parser.add_argument('--deadline-seconds', type=float, default=5400.,
                        help='absolute supervisor bound across all attempts')
    parser.add_argument('--expected-hash', default=None,
                        help='the uninterrupted control sent_hash; the final '
                             'attempt must reproduce it bit-identically')
    parser.add_argument('--python', default=sys_python(),
                        help='interpreter for guarded children (absolute path)')
    parser.add_argument('worker_args', nargs=argparse.REMAINDER,
                        help='worker arguments EXCLUDING --output (supervisor '
                             'injects a fresh one per attempt)')
    args = parser.parse_args()
    if args.worker_args and args.worker_args[0] == '--':
        args.worker_args = args.worker_args[1:]

    # fresh-path refusal: any collision means a previous supervisor run's
    # evidence would be overwritten — refuse with the exact colliding path.
    for n in range(1, args.attempts + 1):
        for kind in ('out', 'guard', 'cell'):
            collision = Path(f'{args.label}-{kind}-{n}.json' if kind != 'cell'
                             else f'{args.label}-cell-{n}.log')
            if collision.exists():
                raise SystemExit(f'refusing: {collision} already exists '
                                f'(retention rule — choose a new --label)')

    report: dict = dict(label=args.label, attempts=[], gpu=args.gpu,
                        expected_hash=args.expected_hash,
                        started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    _atomic_write(f'{args.label}-supervisor.json', report)

    worker_args = args.worker_args
    assert '--output' not in worker_args, 'supervisor injects --output per attempt'
    assert '--inbound-file' in worker_args, \
        'recovery cell must resume from a parked inbound handoff file'
    deadline = time.monotonic() + args.deadline_seconds
    final = None

    for attempt in range(1, args.attempts + 1):
        out = f'{args.label}-out-{attempt}.json'
        guard = f'{args.label}-guard-{attempt}.json'
        cell_log = f'{args.label}-cell-{attempt}.log'
        command = [args.python, '-m', GUARD_MODULE, '--gpu', args.gpu,
                   '--thermal-limit-c', str(args.thermal_limit_c),
                   '--gpu-reserve-mib', str(args.gpu_reserve_mib),
                   '--timeout', str(int(args.guard_timeout)),
                   '--report', guard, '--',
                   args.python, '-m', WORKER_MODULE, '--output', out,
                   *worker_args]
        began = time.monotonic()
        entry: dict = dict(attempt=attempt, launched_at=time.strftime('%H:%M:%S'),
                     guard_report=guard, worker_output=out, cell_log=cell_log)
        report['attempts'].append(entry)
        _atomic_write(f'{args.label}-supervisor.json', report)

        with open(cell_log, 'w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
        entry['child_pid'] = child.pid

        # Death/completion detection: the owned process handle plus the
        # guard report's phase field — never a process-name probe.
        phase = None
        while True:
            phase = _load_guard_phase(guard)
            if phase in DONE_PHASES:
                break
            if child.poll() is not None and phase is None:
                # guard exited without writing a parseable report
                time.sleep(2.)
                phase = _load_guard_phase(guard) or 'guard_exit_no_report'
                break
            if time.monotonic() > deadline:
                child.kill()
                phase = 'supervisor_deadline'
                break
            time.sleep(5.)
        entry['phase'] = phase
        entry['seconds'] = round(time.monotonic() - began, 1)

        verdict = _worker_verdict(out)
        if phase == 'completed' and verdict is not None:
            entry['worker_passed'] = bool(verdict.get('passed'))
            entry['sent_hash'] = verdict.get('sent_hash')
            recovered = (entry['worker_passed']
                         and (args.expected_hash is None
                              or verdict.get('sent_hash') == args.expected_hash))
            entry['hash_matches_control'] = (
                None if args.expected_hash is None
                else bool(verdict.get('sent_hash') == args.expected_hash))
            report['final'] = dict(attempt=attempt, recovered=recovered,
                                   phase=phase,
                                   detection='guard report phase + owned handle')
            _atomic_write(f'{args.label}-supervisor.json', report)
            final = entry
            break
        # failed / killed / no report -> relaunch the SAME cell from the
        # parked inbound file with fresh paths (the recovery under test)
        entry['relaunch'] = True
        _atomic_write(f'{args.label}-supervisor.json', report)
        if time.monotonic() > deadline:
            break

    if final is None:
        report['final'] = dict(recovered=False,
                               reason='attempts exhausted or deadline reached')
        _atomic_write(f'{args.label}-supervisor.json', report)
    print('SUPERVISOR_DONE', json.dumps(report.get('final')))


def sys_python():
    import sys
    return sys.executable


if __name__ == '__main__':
    main()
