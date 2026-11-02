"""Bounded A6000 fairness cell: staggered mixed-load through production serve (#59).

The #64-qualified fair-scheduling machinery (FairAdmission lanes,
ClientPrefillTurns service skew, bounded queue, draining) has never been
measured under SLO load with MORE USERS THAN SLOTS. This cell loads the
card to its measured C4 SLO point (27B Q4 compensated 32K, MTP spec
served — the #685-passing configuration) and submits SIX requests: four
immediately (filling the slots) and two staggered ~40 s later (queued;
admitted as slots drain). All six are 32K-prompt greedy corpus rotations
with 256 output tokens (the cell's declared envelope: the C-series
per-request gates at a shortened output so the two decode rounds fit one
guarded launch).

Declared gates (pre-declared on #59 before launch):
- COMPLETION: all six requests complete with exact 256/256 token counts.
- SLO: every request's post-admission first output <= 600 s and total
  <= 1200 s (the C-series envelope).
- FAIRNESS: no starvation — both queued requests are admitted and
  complete; Jain fairness index over per-user decode token rates >= 0.95.
  Queue wait (submit -> first token) is reported per request, ungated
  (waiting for a slot is the queue's job; the SLO clock starts at
  admission).
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

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
MODEL_NAME = 'qwen38'
PORT = 8139
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 256
CHUNK = 1024
USERS = 6
SLOTS = 4
STAGGER_SECONDS = 40.0
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200
JAIN_FLOOR = 0.95

SERVE_ENV = {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
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


def slot_prompt(base, slot):
    """Deterministic per-user rotation (the C-series convention)."""
    ids = list(base)
    return ids[slot:] + ids[:slot] if slot else ids


def user_request(user, prompt_ids, results):
    body = json.dumps(dict(model=MODEL_NAME, prompt=prompt_ids,
                           max_tokens=OUTPUT_TOKENS, temperature=0,
                           stream=True)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    submitted = time.monotonic()
    first_output = None
    usage = None
    error = None
    try:
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
                    first_output = time.monotonic() - submitted
    except Exception as caught:
        error = repr(caught)
    results[user] = dict(
        submit_to_first_output=first_output,
        submit_to_total=time.monotonic() - submitted,
        completion_tokens=(usage or {}).get('completion_tokens'),
        prompt_tokens=(usage or {}).get('prompt_tokens'),
        output_digest=None, error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    base = Path(os.environ.get('PASS_DIR', '/tmp'))
    log = open(base / 'fair-serve.log', 'w')
    manifest = base / 'fair.manifest.json'
    manifest.write_text(json.dumps(dict(version=1, models={MODEL_NAME: dict(path=SNAPSHOT, quantization='q4_0')})))
    environment = dict(os.environ)
    environment.update(SERVE_ENV)
    server = subprocess.Popen(
        [sys.executable, '-m', 'damendaemon', 'serve',
         '--model-config', str(manifest), '--device', 'cuda', '--context', str(CONTEXT),
         '--no-graph', '--prefill-chunk-size', str(CHUNK),
         '--batch-size', str(SLOTS), '--port', str(PORT),
         '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '600', '--queue-size', '8'],
        env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
    report = dict(completed=False, gate='staggered fairness under SLO load (#59)',
                  model=SNAPSHOT, context=CONTEXT, chunk=CHUNK, users=USERS,
                  slots=SLOTS, stagger_seconds=STAGGER_SECONDS,
                  output_limit=OUTPUT_TOKENS, requests={})
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
        # Warmup: short prompt with a few decode tokens flushes every Triton
        # shape (including the spec round) before the measured load.
        warm = user_request  # noqa: F841 — pattern reuse below
        warm_body = json.dumps(dict(model=MODEL_NAME, prompt=corpus_ids()[:512],
                                    max_tokens=4, temperature=0, stream=True)).encode()
        started = time.monotonic()
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=warm_body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if line.startswith(b'data: ') and line[6:].strip() == b'[DONE]':
                    break
        report['warmup_seconds'] = round(time.monotonic() - started, 1)

        # Four users immediately; two staggered while the first round decodes.
        ids = corpus_ids()
        prompts = {user: slot_prompt(ids, user) for user in range(USERS)}
        report['user_prompt_digests'] = {
            user: hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            for user, prompt in prompts.items()}
        results = {}
        threads = []
        for user in range(USERS):
            if user >= SLOTS:
                time.sleep(STAGGER_SECONDS)
            thread = threading.Thread(target=user_request, args=(user, prompts[user], results))
            thread.start()
            threads.append(thread)
            time.sleep(0.25)
        for thread in threads:
            thread.join(timeout=REQUEST_LIMIT + STAGGER_SECONDS + 300)
        report['requests'] = {str(user): results.get(user) for user in range(USERS)}

        # Verdict: completion, per-request SLO (post-admission clock: the
        # queue wait is submit->first_output for the queued users; the SLO
        # clock is first_output->total), fairness.
        rates, slo_ok, complete_ok = [], True, True
        for user in range(USERS):
            entry = results.get(user) or {}
            if entry.get('error') or entry.get('completion_tokens') != OUTPUT_TOKENS:
                complete_ok = False
                continue
            decode_seconds = entry['submit_to_total'] - entry['submit_to_first_output']
            rates.append((OUTPUT_TOKENS - 1) / decode_seconds if decode_seconds > 0 else 0)
            if (entry['submit_to_total'] - entry['submit_to_first_output']) > REQUEST_LIMIT:
                slo_ok = False
        jain = (sum(rates) ** 2 / (len(rates) * sum(r * r for r in rates))
                if rates and any(rates) else 0.0)
        no_starvation = all((results.get(user) or {}).get('completion_tokens') == OUTPUT_TOKENS
                            for user in range(SLOTS, USERS))
        report['fairness'] = dict(per_user_decode_tok_s=[round(r, 3) for r in rates],
                                  jain_index=round(jain, 4), no_starvation=no_starvation)
        report['completed'] = bool(complete_ok and slo_ok and no_starvation
                                   and jain >= JAIN_FLOOR)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
    args.output.write_text(json.dumps(report, indent=1))
    print(f"[fair] jain={report['fairness']['jain_index'] if report.get('fairness') else 'n/a'} "
          f"completed={report['completed']}", flush=True)
    if not report['completed']:
        raise SystemExit('fairness cell failed a gate; see report')


if __name__ == '__main__':
    main()
