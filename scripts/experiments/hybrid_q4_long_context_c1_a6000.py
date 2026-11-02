"""Bounded A6000 long-context C1 cell: 27B Q4 + MTP spec at 64K/128K (#59/#60).

The #59/#60 context deliverable: qualify 32K minimum upward under the
established envelope. This runner executes ONE bounded C1 attempt per
invocation at a declared --context (65536 or 131072; 256K is refused by
arithmetic — 32 GiB fp32 KV + ~17 GiB packed weights > 44.5 GiB — and is
not launchable). Single user (C1): the context sweep measures the
per-context single-slot envelope; concurrency at 32K is the #685 cell's
delivered result.

Configuration: the #685-passing production serve shape (Q4 compensated,
chunk 1024, MTP head served, greedy corpus prompt at context minus 512
tokens, 512 output). Gates inherited from the C-series: post-admission
first output <= 600 s, complete request <= 1200 s, exact 512/512 usage.
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
PORT = 8140
OUTPUT_TOKENS = 512
CHUNK = 1024
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200

SERVE_ENV = {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
             'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
             'DAMENDAEMON_QUANTIZATION': 'q4_0',
             'DAMENDAEMON_KERNELS': 'triton',
             'DAMENDAEMON_DEVICE': 'cuda',
             'DAMENDAEMON_HYBRID_MTP': '1'}


def corpus_ids(prompt_tokens):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < 2 * prompt_tokens:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:prompt_tokens]


def request(prompt_ids, max_tokens, tag):
    body = json.dumps(dict(model=MODEL_NAME, prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=REQUEST_LIMIT + 300) as response:
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
    result = dict(tag=tag, first_output_seconds=first_output,
                  total_seconds=time.monotonic() - started,
                  output_chars=sum(len(part) for part in text),
                  output_digest=hashlib.sha256(''.join(text).encode()).hexdigest(),
                  usage=usage)
    print(f"[{tag}] first {first_output and round(first_output, 1)}s total "
          f"{round(result['total_seconds'], 1)}s tokens {(usage or {}).get('completion_tokens')}",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(65536, 131072), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    prompt_tokens = args.context - 512
    base = Path(os.environ.get('PASS_DIR', '/tmp'))
    log = open(base / f'ctx{args.context}-serve.log', 'w')
    manifest = base / f'ctx{args.context}.manifest.json'
    manifest.write_text(json.dumps(dict(version=1, models={MODEL_NAME: dict(path=SNAPSHOT, quantization='q4_0')})))
    environment = dict(os.environ)
    environment.update(SERVE_ENV)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT'] = str(args.context)
    server = subprocess.Popen(
        [sys.executable, '-m', 'damendaemon', 'serve',
         '--model-config', str(manifest), '--device', 'cuda',
         '--context', str(args.context), '--no-graph',
         '--prefill-chunk-size', str(CHUNK), '--batch-size', '1',
         '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '600', '--queue-size', '1'],
        env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
    report = dict(completed=False, gate=f'C{1} long-context spec cell (#59/#60)',
                  model=SNAPSHOT, context=args.context, prompt_tokens=prompt_tokens,
                  chunk=CHUNK, output_limit=OUTPUT_TOKENS)
    try:
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
        status = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10).read())
        report['cuda_peak_allocated_bytes_after_warmup'] = status.get('cuda_peak_allocated_bytes')
        # Warmup with a few decode tokens flushes every Triton shape
        # (spec round included) before the measured request.
        request(corpus_ids(512), 4, f'ctx{args.context}-warmup')
        report['round'] = request(corpus_ids(prompt_tokens), OUTPUT_TOKENS,
                                  f'ctx{args.context}-c1')
        final = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10).read())
        report['cuda_peak_allocated_bytes'] = final.get('cuda_peak_allocated_bytes')
        usage = report['round'].get('usage') or {}
        report['completed'] = bool(
            report['round']['first_output_seconds'] is not None
            and report['round']['first_output_seconds'] <= FIRST_OUTPUT_LIMIT
            and report['round']['total_seconds'] <= REQUEST_LIMIT
            and usage.get('completion_tokens') == OUTPUT_TOKENS)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
    args.output.write_text(json.dumps(report, indent=1))
    print(f"[ctx{args.context}] completed={report['completed']}", flush=True)
    if not report['completed']:
        raise SystemExit('context cell failed a gate; see report')


if __name__ == '__main__':
    main()
