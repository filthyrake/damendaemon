"""Bounded real HTTP context-capacity and long-prompt interference proof.

This is an engineering fixture, not model quality, maximum-user qualification,
internal token latency, or a competitive benchmark. Preserve raw SSE timings.
"""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from tokenizers import Tokenizer
if __package__:
    from .benchmark_modern_http import request
else:
    from benchmark_modern_http import request
from dllm.bench import distribution
from dllm.report_io import save_report


@contextmanager
def request_deadlines(first_output, first_seconds, total_seconds, expire):
    """Wall-clock bounds also apply when SSE keeps the socket from timing out."""
    state = {'expired': False, 'reason': None}
    lock = threading.Lock()
    def expired(reason):
        with lock:
            if state['expired']:
                return
            state.update(expired=True, reason=reason)
        expire(reason)
    def first_expired():
        if not first_output.is_set():
            expired('first output')
    timers = [threading.Timer(first_seconds, first_expired),
              threading.Timer(total_seconds, lambda: expired('total request'))]
    for timer in timers:
        timer.daemon = True
        timer.start()
    try:
        yield state
    finally:
        for timer in timers:
            timer.cancel()
        for timer in timers:
            timer.join()


def collect_with_deadlines(collect, first_output, first_seconds, total_seconds, expire):
    """Expiry wins even if graceful server shutdown delivers a complete response."""
    with request_deadlines(first_output, first_seconds, total_seconds, expire) as deadline:
        result = collect()
    # Timer cancellation/join above closes the race with the final verdict.
    if deadline['expired']:
        result = {**result, 'passed': False, 'deadline_exceeded': deadline['reason']}
        result.pop('post_first_output_tokens_per_second_proxy', None)
    return result


def stop_owned_server(process, signum):
    """Signal only the start_new_session server group owned by this runner."""
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def expire_owned_request(process, report, reason):
    report.setdefault('deadline_exceeded', reason)
    # Graceful shutdown can wait for the same HTTP request we must interrupt.
    stop_owned_server(process, signal.SIGKILL)


def check_overflow(base, tokens):
    """Reject one extra reserved output token after the same prompt succeeded."""
    body = dict(model='a', prompt=tokens, max_tokens=513, temperature=0)
    req = Request(base+'/v1/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=30) as response:
            raise AssertionError(f'overflow unexpectedly returned HTTP {response.status}')
    except HTTPError as error:
        payload = json.loads(error.read())
        message = payload.get('error', '')
        assert error.code == 400 and message == 'invalid or unsupported request', payload
        return dict(passed=True, status=error.code, response=payload,
                    prompt_tokens=len(tokens), reserved_output_tokens=513)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--engine-revision', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gap-budget-seconds', type=float, default=1.0)
    parser.add_argument('--capacity-only', action='store_true', help='one near-limit request and unload, without an interference/SLO claim')
    parser.add_argument('--batch-size', type=int, choices=(1, 4), default=4)
    parser.add_argument('--no-graph', action='store_true')
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--experimental-hybrid-context', action='store_true', help='use the isolated issue-68 Q8 context probe')
    probes.add_argument('--experimental-hybrid-q4-context', action='store_true', help='use the isolated issue-68/70 A30/HOMER Q4 context probe')
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--first-output-budget', type=float, default=600)
    parser.add_argument('--host-gib', type=float, help='explicit host-weight residency budget for practical checkpoints')
    args = parser.parse_args()
    if args.context < 32768 or args.chunk_size <= 0 or not 1 <= args.rounds <= 10:
        parser.error('requires context >=32768, positive chunks and 1..10 rounds')
    if not 0 < args.gap_budget_seconds <= 10:
        parser.error('declare a finite short-stream gap budget in (0,10] seconds')
    if any(not math.isfinite(value) or value <= 0 for value in (args.request_timeout, args.first_output_budget)):
        parser.error('request timeout and first-output budget must be positive and finite')
    if args.host_gib is not None and (not math.isfinite(args.host_gib) or args.host_gib <= 0):
        parser.error('host weight budget must be positive and finite')
    if args.batch_size == 1 and not args.capacity_only:
        parser.error('single-slot admission is only valid for --capacity-only')
    if (args.experimental_hybrid_context or args.experimental_hybrid_q4_context) and (not args.no_graph or args.context > 32768):
        parser.error('experimental hybrid context requires --no-graph and context <=32768')
    if args.output.exists():
        parser.error('use a new output path')
    model = Path(args.model)
    config = json.loads((model/'config.json').read_text())
    eos = config.get('text_config', config)['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    tokenizer_path = model/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    size = 2048
    while True:
        ids = tokenizer.encode(''.join(f'Record {i}: The parcel is in cabinet {i%11}, drawer {i%7}.\n'
                                      for i in range(size)), add_special_tokens=False).ids
        if len(ids) >= args.context:
            break
        size *= 2
    short, long = ids[:128], ids[:args.context-512]
    report = dict(passed=False, qualification='context capacity only' if args.capacity_only else 'context/interference engineering fixture only',
        experimental_hybrid_context=args.experimental_hybrid_context, batch_capacity=args.batch_size,
        experimental_hybrid_q4_context=args.experimental_hybrid_q4_context,
        request_timeout=args.request_timeout, first_output_budget=args.first_output_budget,
        host_gib=args.host_gib,
        model_revision=args.model_revision, engine_revision=args.engine_revision,
        context=args.context, output_tokens=512, prefill_chunk_size=args.chunk_size,
        gap_budget_seconds=args.gap_budget_seconds, tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        host=socket.gethostname(), torch=version('torch'),
        environment={key: os.environ.get(key) for key in ('DLLM_DEVICE', 'DLLM_KERNELS',
            'DLLM_QUANTIZATION', 'DLLM_ROCM_MEMORY_GIB', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
            'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'DLLM_HYBRID_PRECISION',
            'DLLM_EXPERIMENTAL_HYBRID_Q4', 'CUDA_VISIBLE_DEVICES')},
        prompts=dict(short=short, long=long), rounds=[],
        limitation='Client visible SSE chunk gaps are not internal per-token latency. Independent numerical and task quality gates are separate.')
    module = 'scripts.experiments.hybrid_context_probe' if args.experimental_hybrid_context else 'dllm'
    if args.experimental_hybrid_q4_context:
        module = 'scripts.experiments.hybrid_q4_context_probe'
    command = [sys.executable, '-m', module, 'serve', '--port', '0', '--context', str(args.context),
        '--prefill-chunk-size', str(args.chunk_size), '--batch-size', str(args.batch_size), '--gpu-models', '1', '--model', 'a='+str(model)]
    if args.no_graph:
        command.append('--no-graph')
    if args.host_gib is not None:
        command.extend(('--host-gib', str(args.host_gib)))
    report['command'] = command
    process = None
    watchdog = None
    cuda_guard = None
    def stop_server(signum):
        stop_owned_server(process, signum)
    def terminate(signum, frame):
        # Stop the owned server before unwinding ThreadPoolExecutor, whose
        # active HTTP readers would otherwise delay cleanup until timeout.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        report.update(passed=False, interrupted_signal=signum, phase='interrupted')
        try:
            save_report(args.output, report)
        finally:
            stop_server(signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        if args.experimental_hybrid_q4_context and os.environ.get('DLLM_DEVICE') == 'cuda':
            from scripts.experiments.cuda_capacity_guard import A30CapacityGuard
            report['cuda_capacity_guard'] = {}
            cuda_guard = A30CapacityGuard(report['cuda_capacity_guard'])
            cuda_guard.read(startup=True)
        report['phase'] = 'starting'
        save_report(args.output, report)
        log_path = args.output.with_suffix('.server.log')
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            if os.environ.get('DLLM_DEVICE') == 'rocm' or cuda_guard is not None:
                from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog
                report['host_memory_guard'] = {}
                guard_options = {'read': cuda_guard.read} if cuda_guard is not None else {}
                watchdog = HostMemoryWatchdog(lambda: stop_server(signal.SIGKILL),
                                              report['host_memory_guard'], **guard_options).start()
            deadline = time.monotonic()+180
            base = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('server exited; see server log')
                for line in log_path.read_text().splitlines():
                    if line.startswith('dllm listening on '):
                        base = line.split()[-1]
                if base:
                    break
                time.sleep(.1)
            if base is None:
                raise TimeoutError('server readiness timeout')
            def run(tokens, first=None):
                visible = first if first is not None else threading.Event()
                def expire(phase):
                    expire_owned_request(process, report, phase)
                return collect_with_deadlines(
                    lambda: request(base, 'a', tokens, 'dllm', eos, threading.Barrier(1),
                                    output_tokens=512, timeout=args.request_timeout, first_output_event=visible),
                    visible, args.first_output_budget, args.request_timeout, expire)
            report['phase'] = 'long_request'
            save_report(args.output, report)
            report['warmup_long'] = run(long)
            assert report['warmup_long']['passed'], 'long warmup failed'
            report['overflow'] = check_overflow(base, long)
            if args.capacity_only:
                result = report['warmup_long']
                report['feasibility_budget_passed'] = (
                    result['first_output_seconds'] <= args.first_output_budget
                    and result['seconds'] <= args.request_timeout)
                assert report['feasibility_budget_passed'], 'capacity request exceeded declared feasibility budget'
                with urlopen(base+'/v1/models/status', timeout=30) as response:
                    report['status_loaded'] = json.load(response)
                assert report['status_loaded']['context'] == args.context
                body = json.dumps(dict(model='a', release_host=True)).encode()
                with urlopen(Request(base+'/v1/models/unload', data=body,
                                     headers={'Content-Type': 'application/json'}), timeout=120) as response:
                    report['unload'] = json.load(response)
                with urlopen(base+'/v1/models/status', timeout=30) as response:
                    report['status_released'] = json.load(response)
                assert report['status_released']['host_cached_bytes'] == 0
                assert all(m['residency'] != 'gpu' for m in report['status_released']['models'])
                report['passed'] = True
                return
            report['control_short'] = run(short)
            assert report['control_short']['passed'], 'short control failed'
            for index in range(args.rounds):
                visible = threading.Event()
                with ThreadPoolExecutor(2) as pool:
                    short_future = pool.submit(run, short, visible)
                    if not visible.wait(180):
                        raise TimeoutError('short request never became visible')
                    if short_future.done():
                        raise RuntimeError('short request finished before interference could begin')
                    long_future = pool.submit(run, long)
                    short_result, long_result = short_future.result(), long_future.result()
                row = dict(index=index, short=short_result, long=long_result, passed=False)
                report['rounds'].append(row)
                assert short_result['passed'] and long_result['passed'], 'request failed'
                row['short_text_matches_control'] = short_result['text'] == report['control_short']['text']
                row['long_text_matches_control'] = long_result['text'] == report['warmup_long']['text']
                times = [short_result['started_monotonic']+c['seconds'] for c in short_result['chunks']]
                start = long_result['started_monotonic']
                stop = start + long_result['first_output_seconds']
                row['short_visible_chunks_during_long_prefill'] = sum(start < t < stop for t in times)
                gaps = [b-a for a,b in zip(times,times[1:])]
                row['short_visible_chunk_gap_seconds'] = distribution(gaps)
                row['max_short_visible_chunk_gap_seconds'] = max(gaps) if gaps else None
                row['gap_budget_passed'] = bool(gaps) and max(gaps) <= args.gap_budget_seconds
                row['passed'] = (row['short_text_matches_control'] and row['long_text_matches_control']
                    and row['short_visible_chunks_during_long_prefill'] >= 2
                    and row['gap_budget_passed'])
                with urlopen(base+'/v1/models/status', timeout=30) as response:
                    row['status_after'] = json.load(response)
                save_report(args.output, report)
                print(f'round {index+1}: passed={row["passed"]} progress_chunks={row["short_visible_chunks_during_long_prefill"]}', flush=True)
            report['passed'] = all(row['passed'] for row in report['rounds'])
    except Exception as error:
        report['error'] = repr(error)
        report['phase'] = 'failed'
        raise
    finally:
        if 'deadline_exceeded' in report:
            report['passed'] = False
        report['phase'] = 'completed' if report['passed'] else report.get('phase', 'failed')
        # Keep a report even if the outer scope kills us during child cleanup.
        try:
            save_report(args.output, report)
        finally:
            if process is not None and process.poll() is None:
                stop_server(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_server(signal.SIGKILL)
                    process.wait()
            report['cleanup_complete'] = process is None or process.poll() is not None
            if watchdog is not None:
                watchdog.close()
                if report['host_memory_guard']['triggered']:
                    report.update(passed=False, phase='host_memory_guard')
            save_report(args.output, report)
            if watchdog is not None and report['host_memory_guard']['triggered']:
                raise SystemExit('host memory reserve exhausted; owned server stopped')
    if not report['passed']:
        raise SystemExit('long-context/interference gate failed; retained report')


if __name__ == '__main__':
    main()
