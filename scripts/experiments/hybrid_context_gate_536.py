"""#536 gate 3: device-qualified 8K (and 32K attempt) context on the A30+A2 pair.

Same guarded two-phase shape as gate 2 (`hybrid_two_rank_gate.py`), but the
pipeline serves at the requested context and the pinned trace sits near the
context limit (context-minus-16 tokens, 32 greedy tokens), so the served cell
actually exercises the KV/state extent being admitted. Capacity and
consistency only: no performance claim (shared guarded GPUs).

Phases (the guard caps a run at 1800 s, so each ~700 s activation runs as
its own guarded invocation):
  --phase two-rank  : pipeline serve at --context, near-limit trace twice.
  --phase single-rank --compare <two-rank report> : single-rank A30 reference
                      of the same trace; finishes the gate on greedy match.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.split_gguf import save

SNAPSHOT = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/'
            'snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
A30_UUID = 'GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a'
A2_UUID = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
PORT_TWO_RANK = 8130
PORT_SINGLE = 8131
BUDGETS_MIB = [22528, 13312]
PREFILL_CHUNK = 127
TRACE_SEED = ('The two-rank context gate replays one pinned raw-token trace '
              'near the admitted context limit so the served cell exercises '
              'the KV extent and the greedy output must stay consistent.')
GREEDY_TOKENS = 32
HOST_FLOOR_BYTES = 100 * 1024**3


def preflight(*, pair_required=True, context):
    if os.environ.get('DAMENDAEMON_SHARED_CUDA_GUARD') != '1':
        raise ValueError('requires execution under scripts.experiments.guard_shared_cuda')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    expected = [A30_UUID, A2_UUID] if pair_required else [A30_UUID]
    if visible != expected:
        raise ValueError(f'gate requires the {"two-device" if pair_required else "single-A30"} '
                         f'guard with the A30 first, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    for uuid, floor in zip(expected, BUDGETS_MIB[:len(expected)]):
        headroom = free.get(uuid)
        if headroom is None or headroom < floor:
            raise ValueError(f'{uuid} headroom {headroom} MiB is below the {floor} MiB stage budget')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable {available} bytes is below the {HOST_FLOOR_BYTES} floor')


def build_trace_ids(context):
    # prompt + greedy output must fit the context: prompt_tokens + max_tokens
    # <= context is the scheduler's admission contract (validate_lengths).
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    ids = tokenizer.encode(TRACE_SEED).ids
    if not ids:
        raise ValueError('pinned trace seed tokenized to zero ids')
    copies = 1
    while len(ids) < context - GREEDY_TOKENS:
        copies *= 2
        ids = tokenizer.encode(TRACE_SEED * copies).ids
    if len(ids) < context - GREEDY_TOKENS:
        raise ValueError('pinned trace seed tokenized below the near-limit trace length')
    return ids[:context - GREEDY_TOKENS]


def gpu_used_mib():
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    return {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}


def status(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def generate(port, prompt_ids, max_tokens, *, read_timeout=1200):
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    chunks = []
    usage = None
    try:
        with urllib.request.urlopen(request, timeout=read_timeout) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if event.get('usage'):
                    usage = event['usage']
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    chunks.append(choices[0]['text'])
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:200]}') from None
    return dict(text=''.join(chunks), usage=usage, seconds=time.monotonic() - started, error=None)


def wait_healthy(port, server, deadline_seconds=900):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return
        except Exception:
            if server.poll() is not None:
                raise RuntimeError('serve exited before becoming healthy') from None
            time.sleep(10)
    raise RuntimeError('serve did not become healthy within 900 seconds')


def serve(environment, port, log_path, manifest_path):
    process = subprocess.Popen(
        [sys.executable, '-u', '-m', 'scripts.experiments.hybrid_q4_context_probe', 'serve',
         '--model-config', str(manifest_path), '--device', 'cuda',
         '--context', str(environment['_DD536_CONTEXT']), '--no-graph', '--batch-size', '1',
         # The #70/#535 cells bound the prefill ramp at chunk 127; an unchunked
         # 8K prefill drives the A2 against the guard's thermal gate.
         '--prefill-chunk-size', str(PREFILL_CHUNK),
         '--port', str(port), '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '600'],
        env=environment, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    try:
        wait_healthy(port, process)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return process


def stop(server):
    if server is None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()


def completion_tokens(result):
    usage = result.get('usage') or {}
    return usage.get('completion_tokens') or 0


def run_two_rank(args, report, trace_ids, manifest_path):
    pipeline_env = dict(os.environ)
    pipeline_env['_DD536_CONTEXT'] = str(args.context)
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    pipeline_env['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
    pipeline_env['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = ','.join(str(v) for v in BUDGETS_MIB)
    server = None
    try:
        server = serve(pipeline_env, PORT_TWO_RANK, args.output.with_suffix('.serve.log'), manifest_path)
        report['memory']['after_load'] = gpu_used_mib()
        report['statuses']['two_rank'] = status(PORT_TWO_RANK)
        report['two_rank']['trace_first'] = generate(PORT_TWO_RANK, trace_ids, GREEDY_TOKENS)
        report['memory']['after_first'] = gpu_used_mib()
        report['two_rank']['trace_repeat'] = generate(PORT_TWO_RANK, trace_ids, GREEDY_TOKENS)
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=f'two-rank phase: {error}')
        save(args.output, report)
        raise
    finally:
        stop(server)
    first = report['two_rank']['trace_first']
    repeat = report['two_rank']['trace_repeat']
    report['two_rank_completed'] = bool(
        completion_tokens(first) == GREEDY_TOKENS
        and completion_tokens(repeat) == GREEDY_TOKENS
        and not first.get('error') and not repeat.get('error')
        and first.get('text') == repeat.get('text'))
    save(args.output, report)


def run_single_rank(args, report, trace_ids, manifest_path):
    reference_env = dict(os.environ)
    reference_env['_DD536_CONTEXT'] = str(args.context)
    reference_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    reference_env['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    for key in ('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE',
                'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT',
                'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'):
        reference_env.pop(key, None)
    server = None
    try:
        server = serve(reference_env, PORT_SINGLE, args.output.with_suffix('.single.log'), manifest_path)
        report['single_rank']['trace'] = generate(PORT_SINGLE, trace_ids, GREEDY_TOKENS)
        report['statuses']['single_rank'] = status(PORT_SINGLE)
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=f'single-rank phase: {error}')
        save(args.output, report)
        raise
    finally:
        stop(server)
    reference = report['single_rank']['trace']
    if args.phase == 'single-rank':
        if not args.compare or not args.compare.exists():
            raise ValueError('single-rank phase requires --compare pointing at the two-rank report')
        with args.compare.open() as source:
            two_rank_report = json.load(source)
        first = two_rank_report.get('two_rank', {}).get('trace_first') or {}
        report['two_rank_report'] = str(args.compare)
        report['two_rank'] = two_rank_report.get('two_rank', {})
        report['two_rank_completed'] = two_rank_report.get('two_rank_completed')
        report['memory'] = two_rank_report.get('memory', {})
        report['statuses']['two_rank'] = two_rank_report.get('statuses', {}).get('two_rank')
    else:
        first = report['two_rank']['trace_first']
    split_text = first.get('text') or ''
    reference_tokens = completion_tokens(reference)
    report['greedy_matches_single_rank'] = bool(
        split_text and reference_tokens == GREEDY_TOKENS and split_text == (reference.get('text') or ''))
    report['completed'] = bool(
        report.get('two_rank_completed')
        and reference_tokens == GREEDY_TOKENS
        and not reference.get('error')
        and report['greedy_matches_single_rank'])
    save(args.output, report)
    print(json.dumps(dict(completed=report['completed'],
                          two_rank_completed=report.get('two_rank_completed'),
                          greedy_matches_single_rank=report['greedy_matches_single_rank'],
                          text=split_text[:80]), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--phase', choices=('two-rank', 'single-rank'), required=True)
    parser.add_argument('--compare', type=Path,
                        help='single-rank phase: path to the two-rank phase report')
    args = parser.parse_args()
    if not 2048 < args.context <= 32768:
        raise ValueError('gate-3 context cells are 2049..32768 (2K is gate 2)')
    preflight(pair_required=args.phase != 'single-rank', context=args.context)
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    trace_ids = build_trace_ids(args.context)
    digest = hashlib.sha256(json.dumps(trace_ids).encode()).hexdigest()
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate=f'#536 gate 3 context {args.context}',
                  model=SNAPSHOT, context=args.context, budgets_mib=BUDGETS_MIB,
                  trace_tokens=len(trace_ids), greedy_tokens=GREEDY_TOKENS,
                  trace_digest=digest, quantization='q4_0', precision='compensated',
                  graphs=False, devices=[A30_UUID, A2_UUID], torch=torch.__version__,
                  two_rank={}, single_rank={}, statuses={}, memory={})
    save(args.output, report)
    if args.phase == 'two-rank':
        run_two_rank(args, report, trace_ids, manifest_path)
    else:
        run_single_rank(args, report, trace_ids, manifest_path)


if __name__ == '__main__':
    main()
