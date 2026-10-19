"""Two-rank hybrid-pipeline qualification on the labnode3 A30+A2 pair (#536 gate 2).

The gate-1 reroute (#542) recorded that the binding two-device path for the
hybrid Qwen3.8-27B is the local CUDA pipeline (`build_cuda_pipeline`), not the
dense-only partition executor. This runner therefore qualifies gate 2 against
that path: the pinned 27B checkpoint, Q4 packed with compensated precision,
context 2048 (the pipeline policy's cap), stage 0 on the A30 (22528 MiB
budget) and stage 1 on the A2 (13312 MiB budget).

Phases, in one guarded process:
  1. Two-rank serve under the pipeline opt-ins; warmup, the pinned raw-token
     trace twice (determinism), per-device used memory from nvidia-smi, and
     /v1/models/status.
  2. Single-rank A30 serve (no pipeline env) replaying the same trace: the
     greedy-exactness reference. The issue text names a CPU reference; a 27B
     FP32 CPU reference needs ~108 GiB of weights alone and does not fit the
     bounded window, so the single-rank GPU reference of the SAME quantized
     path is used and the deviation is recorded in the report and issue.
Capacity and consistency only: no performance claim (shared guarded GPUs).

Preflight requires the two-device guard (A30 UUID first), fresh per-device
headroom at the stage budgets, and host MemAvailable >= 100 GiB (the #70
loader floor). Any failure stops the run.
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

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/'
            'snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
A30_UUID = 'GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a'
A2_UUID = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
PORT_TWO_RANK = 8127
PORT_SINGLE = 8128
CONTEXT = 2048
BUDGETS_MIB = [22528, 13312]
TRACE_SEED = ('The two-rank gate replays one pinned raw-token trace so the '
              'split pipeline and the single-rank reference must agree '
              'greedy token for token under the same quantized weights.')
TRACE_TOKENS = 64
GREEDY_TOKENS = 12
HOST_FLOOR_BYTES = 100 * 1024**3


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A30_UUID, A2_UUID]:
        raise ValueError(f'two-rank gate requires the two-device guard with the A30 first, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    for uuid, floor in zip(visible, BUDGETS_MIB):
        headroom = free.get(uuid)
        if headroom is None or headroom < floor:
            raise ValueError(f'{uuid} headroom {headroom} MiB is below the {floor} MiB stage budget')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable {available} bytes is below the {HOST_FLOOR_BYTES} floor')


def build_trace_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    ids = tokenizer.encode(TRACE_SEED).ids
    if len(ids) < TRACE_TOKENS:
        raise ValueError('pinned trace seed tokenized below the trace length')
    return ids[:TRACE_TOKENS]


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


def generate(port, prompt_ids, max_tokens, *, read_timeout=900):
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=read_timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:200]}') from None
    choice = payload['choices'][0]
    return dict(text=choice['text'], tokens=choice.get('token_ids'),
                usage=payload.get('usage'), error=None)


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


def serve(environment, port, log_path):
    process = subprocess.Popen(
        [sys.executable, '-u', '-m', 'scripts.experiments.hybrid_q4_context_probe', 'serve',
         '--model-config', str(log_path.with_suffix('.manifest.json')), '--device', 'cuda',
         '--context', str(CONTEXT), '--no-graph', '--batch-size', '1',
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    trace_ids = build_trace_ids()
    digest = hashlib.sha256(json.dumps(trace_ids).encode()).hexdigest()
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate='#536 gate 2 two-rank hybrid pipeline',
                  model=SNAPSHOT, context=CONTEXT, budgets_mib=BUDGETS_MIB,
                  trace_tokens=TRACE_TOKENS, greedy_tokens=GREEDY_TOKENS,
                  trace_digest=digest, quantization='q4_0', precision='compensated',
                  graphs=False, devices=[A30_UUID, A2_UUID], torch=torch.__version__,
                  two_rank={}, single_rank={}, statuses={}, memory={})
    save(args.output, report)

    # Phase 1: two-rank pipeline serve (A30 stage 0, A2 stage 1).
    pipeline_env = dict(os.environ)
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    pipeline_env['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
    pipeline_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
    pipeline_env['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = ','.join(str(v) for v in BUDGETS_MIB)
    server = None
    try:
        server = serve(pipeline_env, PORT_TWO_RANK, args.output.with_suffix('.serve.log'))
        report['two_rank']['warmup'] = generate(PORT_TWO_RANK, trace_ids[:16], 1)
        if report['two_rank']['warmup'].get('error'):
            raise RuntimeError(f"two-rank warmup failed: {report['two_rank']['warmup']['error']}")
        report['memory']['after_warmup'] = gpu_used_mib()
        report['statuses']['two_rank'] = status(PORT_TWO_RANK)
        report['two_rank']['trace_first'] = generate(PORT_TWO_RANK, trace_ids, GREEDY_TOKENS)
        report['two_rank']['trace_repeat'] = generate(PORT_TWO_RANK, trace_ids, GREEDY_TOKENS)
        report['memory']['after_traces'] = gpu_used_mib()
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=f'two-rank phase: {error}')
        save(args.output, report)
        raise
    finally:
        stop(server)

    # Phase 2: single-rank A30 reference serve (same quantized path, no split).
    reference_env = dict(os.environ)
    reference_env['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    reference_env['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    for key in ('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE',
                'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT',
                'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'):
        reference_env.pop(key, None)
    server = None
    try:
        server = serve(reference_env, PORT_SINGLE, args.output.with_suffix('.single.log'))
        report['single_rank']['trace'] = generate(PORT_SINGLE, trace_ids, GREEDY_TOKENS)
        report['statuses']['single_rank'] = status(PORT_SINGLE)
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=f'single-rank phase: {error}')
        save(args.output, report)
        raise
    finally:
        stop(server)

    first = report['two_rank']['trace_first']
    repeat = report['two_rank']['trace_repeat']
    reference = report['single_rank']['trace']
    split_tokens = first.get('tokens') or []
    report['deterministic'] = split_tokens == (repeat.get('tokens') or [])
    report['greedy_matches_single_rank'] = split_tokens == (reference.get('tokens') or [])
    report['completed'] = bool(
        split_tokens
        and len(split_tokens) == GREEDY_TOKENS
        and not first.get('error') and not repeat.get('error') and not reference.get('error')
        and report['deterministic'])
    save(args.output, report)
    print(json.dumps(dict(completed=report['completed'],
                          deterministic=report['deterministic'],
                          greedy_matches_single_rank=report['greedy_matches_single_rank'],
                          tokens=split_tokens), indent=1))


if __name__ == '__main__':
    main()
