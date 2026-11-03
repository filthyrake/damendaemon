"""Spec-round CUDA-graph A/B: eager MTP decode vs graph replay (#691).

The #684 A/B qualified the 27B MTP spec serve on plain eager decode —
the eager-MTP constraint refused CUDA graphs because the packed spec
round ([B, 2] divergent-position continuations) could not ride the
static batch decode graphs. #691 lifts that constraint for the spec
round's main forward: the round is captured per batch size as a CUDA
graph over a NEW static full-context window grouped attention (the
positions matrix carries every dynamic degree of freedom; no [:end]
slicing, no host reads inside capture), with the MTP proposal/backfill
staying eager between replays.

This runner measures the C1 cell (the #684 record shape): single 32K
request, 32,256-token pinned prompt, 512 greedy output tokens, chunk
1024, 27B Q4 compensated spec serve, one guarded invocation per phase:

- phase eager : DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH=0, --no-graph — the
                qualified #684 shape (the control; the on-record decode
                is 54.8 s / 9.3 tok/s for 512 tokens).
- phase graph : DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH=1, graphs on — the
                probe's isolated qualification view admits CUDA graphs
                for this surface only under the opt-in; production
                hybrid_precision.select is unchanged.

Gates (declared on #691 before launch):
- COMPLETENESS: both phases complete 512/512 tokens, zero errors.
- EXACTNESS: the graph phase's output digest equals the eager phase's
  byte-for-byte (greedy semantics; graph replay must not change the
  result — any divergence is an integration defect).
- DECODE: the graph phase's decode window (completion wall minus first
  output) is strictly shorter than the eager phase's, measured in the
  same run with the same pinned corpus; the ratio is reported. The
  eager phase must reproduce the #684 record's magnitude (decode window
  30-80 s) for the A/B to count as the baseline control.
Prefill first-output is reported for both phases but not gated (the
lever is decode-side only; identical scan prefill runs in both).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8150
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
CHUNK = 1024
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
PINNED_PROMPT_DIGEST = ('31836e963e26cd642684d7bce9c7781323768'
                        '00375c2b09917612a5d398be727')
EAGER_DECODE_FLOOR = 30.0
EAGER_DECODE_CEILING = 80.0


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


def build_prompt_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    if not ids:
        raise ValueError('prompt paragraph tokenized to zero ids')
    copies = 1
    while len(ids) < 2 * PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    if len(ids) < 2 * PROMPT_TOKENS:
        raise ValueError('prompt text did not tokenize to the required length')
    return ids[:PROMPT_TOKENS]


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def request(prompt_ids, max_tokens, tag, socket_timeout):
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    chunks = []
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
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    chunks.append(choices[0]['text'])
    except Exception as failure:
        error = failure
    text = ''.join(chunks)
    total = time.monotonic() - started
    decode_window = (total - first_output) if first_output is not None else None
    result = dict(tag=tag, first_output_seconds=first_output, total_seconds=total,
                  decode_window_seconds=decode_window, output_chars=len(text),
                  output_ids_digest=hashlib.sha256(json.dumps(text).encode()).hexdigest(),
                  usage=usage, error=str(error) if error else None)
    if text:
        result['output_preview'] = text[:80]
    print(f"[{tag}] first {first_output and round(first_output, 1)}s total {round(total, 1)}s "
          f"decode {decode_window and round(decode_window, 1)}s "
          f"tokens {(usage or {}).get('completion_tokens')} error={bool(error)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('eager', 'graph'), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    base = build_prompt_ids()
    prompt_digest = hashlib.sha256(json.dumps(base).encode()).hexdigest()
    if prompt_digest != PINNED_PROMPT_DIGEST:
        raise ValueError('prompt digest does not match the pinned C-series corpus; refusing to launch')
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate='#691 spec-round CUDA graph A/B (C1, 27B spec serve)',
                  phase=args.phase, model=SNAPSHOT,
                  model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS, output_limit=OUTPUT_TOKENS,
                  chunk=CHUNK, batch=1, quantization='q4_0', precision='compensated',
                  spec_graph=(args.phase == 'graph'), device=A6000_UUID,
                  prompt_digest=prompt_digest, rounds={}, statuses={})
    save(args.output, report)
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    environment['DAMENDAEMON_HYBRID_MTP'] = '1'
    environment['DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH'] = '1' if args.phase == 'graph' else '0'
    command = [sys.executable, '-m', 'scripts.experiments.hybrid_q4_a6000_probe', 'serve',
               '--model-config', str(manifest_path), '--device', 'cuda', '--context', str(CONTEXT),
               '--prefill-chunk-size', str(CHUNK),
               '--batch-size', '1', '--port', str(PORT),
               '--gpu-models', '1', '--host-models', '1', '--host-gib', '40',
               '--queue-timeout', '600', '--queue-size', '1']
    if args.phase == 'eager':
        # The qualified #684 shape: graphs off at the executor level too.
        command.append('--no-graph')
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
        warm = request(base[:512], 4, f'{args.phase}-warmup', socket_timeout=900)
        report['rounds']['warmup'] = warm
        report['statuses']['after_warmup'] = status()
        save(args.output, report)
        if warm['error']:
            raise RuntimeError(f"warmup failed: {warm['error']}")
        report['rounds']['c1'] = request(base, OUTPUT_TOKENS, f'{args.phase}-32k', socket_timeout=1500)
        report['statuses']['after_round'] = status()
        save(args.output, report)
        round_result = report['rounds']['c1']
        tokens_ok = (round_result.get('usage') or {}).get('completion_tokens') == OUTPUT_TOKENS
        if round_result['error'] or not tokens_ok:
            raise RuntimeError(f"cell incomplete: error={round_result['error']} tokens_ok={tokens_ok}")
        decode = round_result['decode_window_seconds']
        if args.phase == 'eager':
            # The control must reproduce the #684 record's magnitude for
            # the A/B to count as the baseline (54.8 s on record).
            report['eager_control_confirmed'] = bool(
                decode is not None and EAGER_DECODE_FLOOR <= decode <= EAGER_DECODE_CEILING)
            report['completed'] = report['eager_control_confirmed']
        else:
            if decode is None:
                raise RuntimeError('graph phase produced no decode window')
            report['completed'] = True
        save(args.output, report)
        print(f"[{args.phase}] decode window {round(decode, 1)}s digest "
              f"{round_result['output_ids_digest'][:16]} completed={report['completed']}", flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
    if not report['completed']:
        raise SystemExit('phase failed a gate; see report')


if __name__ == '__main__':
    main()
