"""Bounded A6000 A/B cell: MTP speculative SERVING vs plain decode (#68).

The #683 cell qualified the 27B's dense MTP draft head on the executor
API (12/12 exact, embed_first acceptance 71% short / 100% on the pinned
corpus continuation). This cell measures the SERVING integration — the
scheduler's packed spec rounds (one [B, 2] optimistic forward per round,
up to two verified tokens per slot, per-slot state rollback on rejection,
chunked-prefill coexistence via the incremental MTP KV fill) — on plain
production `damendaemon serve`.

A/B in two separately guarded phases (the 27B's decode times do not fit
both passes in one 1800 s guard): identical pinned workload (Q4
compensated, 32K context, 32,256-token prompt, 512 output, chunk 1024 —
the qualified C2 setting), pass 'plain' with the head dropped, pass
'spec' with the head loaded (the scheduler engages spec rounds
structurally for greedy requests).

Declared gates (pre-declared on #68 before launch):
- EXACTNESS: the spec pass's output text must equal the plain pass's
  byte-for-byte (same corpus, same greedy semantics — spec is exact by
  construction; any divergence is an integration defect).
- THROUGHPUT: the spec pass's 512-token decode must complete in <= 0.85x
  the plain pass's decode wall-clock. The pinned corpus accepts ~100%
  (measured 511/511 in #683), so rounds emit two tokens per forward — a
  structural ~2x decode bound; 0.85x is the conservative gate. The
  measured ratio is reported; no claim beyond it.
Prefill first-output is reported for both passes but not gated (the lever
is decode-side; the same grouped prefill runs in both).
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
PORT = 8137
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
CHUNK = 1024
THROUGHPUT_RATIO = 0.85

PHASE_ENV = {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
             'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
             'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': str(CONTEXT),
             'DAMENDAEMON_QUANTIZATION': 'q4_0',
             'DAMENDAEMON_KERNELS': 'triton',
             'DAMENDAEMON_DEVICE': 'cuda',
             'DAMENDAEMON_HYBRID_MTP': '1'}


def corpus_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < 2 * PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:PROMPT_TOKENS]


def request(prompt_ids, max_tokens, tag):
    body = json.dumps(dict(model=MODEL_NAME, prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=1500) as response:
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
          f"{round(result['total_seconds'], 1)}s chars {result['output_chars']}", flush=True)
    return result


def phase(args):
    """One guarded phase: boot production serve (head loaded or dropped),
    warm up, run the pinned 32K/512 request, tear down."""
    environment = dict(os.environ)
    environment.update(PHASE_ENV)
    if args.phase == 'plain':
        environment['DAMENDAEMON_HYBRID_MTP_DROP'] = '1'
    base = Path(os.environ.get('PASS_DIR', '/tmp'))
    log = open(base / f'mtp-serve-{args.phase}.log', 'w')
    manifest = base / f'mtp-serve-{args.phase}.manifest.json'
    manifest.write_text(json.dumps(dict(version=1, models={MODEL_NAME: dict(path=SNAPSHOT, quantization='q4_0')})))
    server = subprocess.Popen(
        [sys.executable, '-m', 'damendaemon', 'serve',
         '--model-config', str(manifest), '--device', 'cuda', '--context', str(CONTEXT),
         '--no-graph', '--prefill-chunk-size', str(CHUNK),
         '--batch-size', '1', '--port', str(PORT),
         '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '600', '--queue-size', '1'],
        env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
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
        # The draft head's packed weight delta is the load-detection control:
        # the spec phase must carry ~0.65 GiB more model bytes than plain.
        host_weight_bytes = sum(model.get('host_weight_bytes') or 0
                                for model in status.get('models', []))
        # Warmup: a short prompt with a few output tokens flushes Triton JIT
        # for every serve shape — including the spec round's [1, 2]
        # continuation forward — before timing (max_tokens 1 would finish
        # at the prefill token and never reach a decode round).
        warm = request(corpus_ids()[:512], 4, f'{args.phase}-warmup')
        result = dict(phase=args.phase, model=SNAPSHOT, context=CONTEXT,
                      chunk=CHUNK, host_weight_bytes=host_weight_bytes, warmup=warm,
                      cuda_peak_allocated_bytes=status.get('cuda_peak_allocated_bytes'))
        result['round'] = request(corpus_ids(), OUTPUT_TOKENS, f'{args.phase}-32k')
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
    args.output.write_text(json.dumps(result, indent=1))
    print(f"[{args.phase}] completed; digest {result['round']['output_digest'][:16]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('plain', 'spec'), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    phase(args)


if __name__ == '__main__':
    main()
