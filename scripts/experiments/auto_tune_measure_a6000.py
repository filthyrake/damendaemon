"""#674 cell 3: measure ONE declared candidate under the guard (A6000).

Consumes a tune manifest (cell 2), enumerates the manifest's own declared
search (cell 1's `declared_grid` + `enumerate_candidates` — production
policy only, no experimental opt-ins), and measures exactly one candidate
per guarded invocation: the serve boots with the candidate's environment
and flags, a warmup request flushes JIT for the shape, then the manifest's
pinned workload runs (digest-verified corpus, exact counts, declared
concurrency). The row is written incrementally (retained on crash) as a
cell-2 `Measurement`-shaped JSON with the candidate and both objective
metrics, ready for `auto_tune_rank`.

One candidate per invocation is deliberate: environment/flag changes are
process-level, so a candidate IS a serve boot — the sweep is a sequence of
guarded invocations (the #691/#690 pattern), never a long-lived tuner.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from damendaemon.autotune import (declared_grid, enumerate_candidates,
                                  parse_manifest, serve_manifest)
from damendaemon.models import config as model_config
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8152
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
WARMUP_PROMPT_TOKENS = 512
WARMUP_OUTPUT_TOKENS = 4
C_SERIES_PARAGRAPH = ('The quick brown fox jumps over the lazy dog while the '
                      'repository records every bounded qualification cell '
                      'with evidence and pins. ')


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'gate requires the single-device A6000 guard, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    headroom = free.get(A6000_UUID)
    if headroom is None or headroom < GPU_FLOOR_MIB:
        raise ValueError(f'A6000 headroom {headroom} MiB is below the {GPU_FLOOR_MIB} MiB floor')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable {available} bytes is below the {HOST_FLOOR_BYTES} floor')


def build_pinned_corpus(model_path, prompt_tokens, pinned_digest):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(model_path) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    ids = tokenizer.encode(C_SERIES_PARAGRAPH).ids
    if not ids:
        raise ValueError('prompt paragraph tokenized to zero ids')
    copies = 1
    while len(ids) < 2 * prompt_tokens:
        copies *= 2
        ids = tokenizer.encode(C_SERIES_PARAGRAPH * copies).ids
    ids = ids[:prompt_tokens]
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if digest != pinned_digest:
        raise ValueError('the pinned corpus digest does not match this manifest; '
                         'refusing to measure against a different workload')
    return ids


def has_mtp_tensors(path):
    """Cheap checkpoint probe: MTP tensors in the safetensors index —
    never a weight load. GGUF checkpoints refuse the probe explicitly
    (the manifest's mtp field must be verified by hand for those)."""
    index = Path(path) / 'model.safetensors.index.json'
    if index.is_file():
        keys = json.loads(index.read_text()).get('weight_map', {})
        return any(name.startswith('mtp.') for name in keys)
    from damendaemon.models import is_gguf
    if is_gguf(path):
        raise ValueError('GGUF MTP probing is not implemented; verify the '
                         'checkpoint by hand before declaring mtp in the manifest')
    return False


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def request(prompt_ids, max_tokens, tag, results, socket_timeout):
    body = json.dumps(dict(model='tune', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    usage = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=socket_timeout) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if first_output is None and (event.get('choices') or [{}])[0].get('text'):
                    first_output = time.monotonic() - started
                if event.get('usage'):
                    usage = event['usage']
    except Exception as failure:
        error = failure
    results[tag] = dict(first_output_seconds=first_output,
                        request_seconds=time.monotonic() - started,
                        completion_tokens=(usage or {}).get('completion_tokens'),
                        error=str(error) if error else None)


def measure_round(prompts, max_tokens, concurrency, socket_timeout):
    """Run the manifest's request count at the declared concurrency."""
    results = {}
    threads = []
    batch = min(concurrency, len(prompts))
    for start in range(0, len(prompts), batch):
        group = {f'r{index}': prompts[index]
                for index in range(start, min(start + batch, len(prompts)))}
        threads = [threading.Thread(target=request,
                                   args=(ids, max_tokens, tag, results, socket_timeout))
                   for tag, ids in group.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return results


def row_from_results(candidate, results, round_seconds, metric):
    first = [row['first_output_seconds'] for row in results.values()
             if row['first_output_seconds'] is not None]
    seconds = [row['request_seconds'] for row in results.values()]
    tokens = [row['completion_tokens'] for row in results.values()
              if row['completion_tokens'] is not None]
    errors = sum(1 for row in results.values() if row['error'])
    successful_tokens = sum(tokens) if not errors else None
    aggregate = (successful_tokens / round_seconds
                 if successful_tokens and round_seconds > 0 else None)
    p95 = (sorted(first)[max(0, int(round(len(first) * 0.95)) - 1)]
           if first else None)
    return dict(
        candidate=candidate.__dict__,
        metric_value=(p95 if metric == 'p95_first_output_seconds' else aggregate),
        p95_first_output_seconds=p95,
        aggregate_output_tokens_per_second=aggregate,
        first_output_seconds=first,
        request_seconds=seconds,
        completion_tokens=tokens,
        errors=errors,
        requests_measured=len(results),
        unmeasurable_reason=(None if not errors and tokens else
                             'errors or missing usage in the measured round'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--candidate', type=int, required=True,
                        help='index into the manifest\'s enumerated admitted candidates')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    manifest = parse_manifest(args.manifest.read_bytes())
    model = manifest['model']
    config = model_config(model['path'])
    has_mtp = has_mtp_tensors(model['path'])
    # model.mtp is the SEARCH intent (may the sweep include head-armed
    # candidates); the checkpoint probe validates that intent — it does
    # not require the head (the qualified headless serve drops it).
    declared = manifest['search'].get('values', {})
    declared_mtp_on = 'on' in declared.get('mtp', [])
    if declared_mtp_on != model['mtp']:
        raise ValueError('manifest model.mtp must match the declared mtp '
                         'search values (the field is the search intent, '
                         'not the checkpoint contents)')
    if model['mtp'] and not has_mtp:
        raise ValueError('the search declares MTP candidates but the '
                         'checkpoint carries no MTP tensors')
    grid = declared_grid(manifest['search'].get('values', {}), model['quantization'])
    admitted, refusals = enumerate_candidates(
        config, accelerator=manifest['hardware']['accelerator'],
        context=manifest['context'], has_mtp=has_mtp, grid=grid)
    if not admitted:
        raise RuntimeError(f'the manifest\'s declared search admits no candidates; '
                           f'refusals: {[(str(c), r) for c, r in refusals][:5]}')
    if not 0 <= args.candidate < len(admitted):
        raise ValueError(f'candidate index {args.candidate} outside the admitted '
                         f'set (0..{len(admitted) - 1})')
    candidate = admitted[args.candidate]
    requests_spec = manifest['requests']
    prompts = [build_pinned_corpus(model['path'], manifest['corpus']['prompt_tokens'],
                                   manifest['corpus']['prompt_sha256'])]
    # Distinct prompts per request slot (the C-series rotation), pinned by digest.
    base = prompts[0]
    for slot in range(1, requests_spec['count']):
        rotated = (base[slot:] + base[:slot])[:manifest['corpus']['prompt_tokens']]
        if rotated == base:
            raise ValueError(f'slot {slot} rotation collided with slot 0')
        prompts.append(rotated)

    row = dict(candidate=candidate.__dict__, candidate_index=args.candidate,
               objective=manifest['objective'], completed=False,
               gates=manifest['objective']['gates'])
    save(args.output, row)
    environment = dict(os.environ)
    environment.update(candidate.environment())
    manifest_path = args.output.with_suffix('.model-config.json')
    save(manifest_path, serve_manifest('tune', model['path'], candidate.quantization))
    command = [sys.executable, '-m', 'damendaemon', 'serve',
               '--model-config', str(manifest_path), '--device', 'cuda',
               '--context', str(manifest['context']),
               *candidate.serve_flags(),
               '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
               '--host-gib', '40', '--queue-timeout', '600',
               '--queue-size', str(requests_spec['concurrency'])]
    server = None
    try:
        server = subprocess.Popen(command, env=environment,
                                  stdout=open(args.output.with_suffix('.serve.log'), 'w'),
                                  stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f'serve exited {server.returncode} during warmup')
                time.sleep(5)
        else:
            raise RuntimeError('serve never became healthy')
        warm = {}
        request(base[:WARMUP_PROMPT_TOKENS], WARMUP_OUTPUT_TOKENS, 'warmup', warm, 900)
        if warm['warmup']['error']:
            raise RuntimeError(f"warmup failed: {warm['warmup']['error']}")
        row['warmup'] = warm['warmup']
        row['status_after_warmup'] = status()
        save(args.output, row)

        round_started = time.monotonic()
        results = measure_round(prompts, requests_spec['max_tokens'],
                                requests_spec['concurrency'],
                                socket_timeout=1500)
        round_seconds = time.monotonic() - round_started
        row.update(row_from_results(candidate, results, round_seconds,
                                     manifest['objective']['metric']))
        row['status_after_round'] = status()
        row['round_seconds'] = round_seconds
        row['completed'] = True
        save(args.output, row)
        print(f"[candidate {args.candidate}] metric {row['metric_value']} "
              f"p95_first {row['p95_first_output_seconds']} errors {row['errors']}",
              flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
    if not row['completed']:
        raise SystemExit('measurement did not complete; see row')


if __name__ == '__main__':
    main()
