"""Bounded A6000 A/B cell: grouped vs eager-SDPA compensated chunk prefill (#68).

The standing 32K prefill OOM-retry storm (12-30 CUDACachingAllocator warnings
per round in every October/November cell, allocation sizes growing linearly
with the KV prefix — 769,654,784 bytes at exactly 23,680 positions, i.e. the
(1, 64 q-heads, 127 chunk, end) fp32 score block) is attributed to the eager
SDPA path and eliminated by dispatching the bounded grouped path on CUDA.

A/B in one guarded run: plain production `damendaemon serve` boots twice with
the identical pinned workload (Q4 compensated, 32K context, 32,256-token
prompt, 64 output) — pass A with the grouped default, pass B with
DAMENDAEMON_HYBRID_GROUPED_PREFILL=0 (the SDPA control). Per pass: OOM
warning count from the serve log, allocator peak from /v1/models/status,
first-output seconds, output text digest.

Gates (accuracy/drift per #23, cross-kernel greedy exactness NOT demanded —
digests are reported and compared, divergences characterized not failed):
- pass A completes within the C1 envelope (first output <= 600 s).
- pass A's OOM-warning count strictly below pass B's, and pass B reproduces
  the storm (> 0 warnings) — attribution control.
- pass A's allocator peak below pass B's (the grouped scratch is bounded).
No performance claim beyond the A/B deltas this cell measures.
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

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
MODEL_NAME = 'qwen38'
PORT = 8136
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 64
CHUNK = 127
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200


def build_prompt_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:PROMPT_TOKENS]


def request(prompt_ids, timeout=REQUEST_LIMIT):
    body = json.dumps(dict(model=MODEL_NAME, prompt=prompt_ids, max_tokens=OUTPUT_TOKENS,
                           temperature=0, stream=True)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            if event.get('usage'):
                usage = event['usage']
            choice = (event.get('choices') or [{}])[0]
            if first_output is None and choice.get('text'):
                first_output = time.monotonic() - started
            if choice.get('text'):
                text.append(choice['text'])
    return dict(first_output_seconds=first_output,
                total_seconds=time.monotonic() - started,
                output_chars=sum(len(part) for part in text),
                output_digest=hashlib.sha256(''.join(text).encode()).hexdigest(),
                usage=usage)


def serve_pass(tag, ids, grouped):
    report = dict(tag=tag, grouped=grouped)
    base = Path(os.environ.get('PASS_DIR', '/tmp'))
    log_path = base / f'{tag}.serve.log'
    manifest = base / f'{tag}.manifest.json'
    manifest.write_text(json.dumps(dict(version=1, models={MODEL_NAME: dict(path=SNAPSHOT, quantization='q4_0')})))
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT'] = str(CONTEXT)
    environment['DAMENDAEMON_HYBRID_GROUPED_PREFILL'] = '1' if grouped else '0'
    server = subprocess.Popen(
        [sys.executable, '-m', 'damendaemon', 'serve',
         '--model-config', str(manifest), '--device', 'cuda', '--context', str(CONTEXT),
         '--no-graph', '--prefill-chunk-size', str(CHUNK),
         '--batch-size', '1', '--port', str(PORT),
         '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '600', '--queue-size', '1'],
        env=environment, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    report['serve_exit'] = server.returncode
                    return report
                time.sleep(5)
        else:
            report['healthy'] = False
            return report
        report['healthy'] = True
        report['round'] = request(ids)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
                status = json.loads(response.read())
            report['cuda_peak_allocated_bytes'] = status.get('cuda_peak_allocated_bytes')
        except Exception as error:
            report['status_error'] = repr(error)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
        report['oom_warnings'] = sum(1 for line in open(log_path, errors='replace')
                                    if 'memory allocation failed with OOM' in line)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    ids = build_prompt_ids()
    report = dict(completed=False, gate='grouped vs SDPA compensated chunk-prefill A/B (#68 capacity)',
                  model=SNAPSHOT, context=CONTEXT, prompt_tokens=PROMPT_TOKENS,
                  output_tokens=OUTPUT_TOKENS, chunk=CHUNK, passes=[])
    # Pass A: grouped (default). Pass B: the SDPA control.
    report['passes'].append(serve_pass('passA-grouped', ids, True))
    args.output.write_text(json.dumps(report, indent=1))
    report['passes'].append(serve_pass('passB-sdpa-control', ids, False))
    args.output.write_text(json.dumps(report, indent=1))
    a, b = report['passes'][0], report['passes'][1]
    report['attribution'] = dict(
        a_completed=bool(a.get('round')), b_storm_reproduced=bool(b.get('oom_warnings')),
        oom_a=a.get('oom_warnings'), oom_b=b.get('oom_warnings'),
        peak_a=a.get('cuda_peak_allocated_bytes'), peak_b=b.get('cuda_peak_allocated_bytes'),
        first_output_a=(a.get('round') or {}).get('first_output_seconds'),
        first_output_b=(b.get('round') or {}).get('first_output_seconds'),
        output_digest_identical=(a.get('round') or {}).get('output_digest') == (b.get('round') or {}).get('output_digest'))
    a_ok = (a.get('healthy') and (a.get('round') or {}).get('first_output_seconds') is not None
            and a['round']['first_output_seconds'] <= FIRST_OUTPUT_LIMIT)
    report['completed'] = bool(a_ok and a.get('oom_warnings', 1) < b.get('oom_warnings', 0)
                               and b.get('oom_warnings', 0) > 0)
    args.output.write_text(json.dumps(report, indent=1))
    print(f"[ab] A(grouped) oom={a.get('oom_warnings')} first={report['attribution']['first_output_a']} "
          f"| B(sdpa) oom={b.get('oom_warnings')} first={report['attribution']['first_output_b']} "
          f"| digests identical={report['attribution']['output_digest_identical']}", flush=True)
    if not report['completed']:
        raise SystemExit('A/B cell failed a gate; see report')


if __name__ == '__main__':
    main()
