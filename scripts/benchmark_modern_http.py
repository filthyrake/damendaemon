"""Fixed modern-model HTTP workload; measurements are not a parity verdict.

Run against an isolated benchmark server. Model/engine/precision pins belong in
the supplied manifest. The caller owns sequential ABBA process orchestration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tokenizers import Tokenizer
from dllm.bench import distribution, events, llama_context
from dllm.report_io import save_report


MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_FRAGMENTS = 100_000


def request(base, model, tokens, engine, eos_ids, barrier, *, output_tokens=128, timeout=180,
            first_output_event=None):
    body = dict(model=model, prompt=tokens, temperature=0, max_tokens=output_tokens,
                stream=True, stream_options={'include_usage': True},
                logit_bias={str(token): -100 for token in eos_ids})
    if engine == 'llama.cpp':
        body.update(cache_prompt=False)
    start = time.perf_counter()
    result = dict(prompt_tokens=len(tokens), passed=False, chunks=[], started_monotonic=start,
                  start_barrier_passed=False,
                  capture_limits=dict(utf8_bytes=MAX_CAPTURE_BYTES, fragments=MAX_CAPTURE_FRAGMENTS))
    text = []
    captured_bytes = 0
    try:
        barrier.wait(timeout=30)
        start = time.perf_counter()
        result.update(started_monotonic=start, start_barrier_passed=True)
        req = Request(base + '/v1/completions', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
        done = False
        with urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('expected SSE')
            for data in events(response, require_terminated=True):
                elapsed = time.perf_counter() - start
                if data == '[DONE]':
                    done = True
                    break
                item = json.loads(data)
                if item.get('error'):
                    raise ValueError(str(item['error']))
                for choice in item.get('choices', []):
                    fragment = choice.get('text', '')
                    if not isinstance(fragment, str):
                        raise ValueError('output fragment must be text')
                    if fragment:
                        size = len(fragment.encode('utf-8'))
                        if captured_bytes + size > MAX_CAPTURE_BYTES or len(text) >= MAX_CAPTURE_FRAGMENTS:
                            raise ValueError('benchmark output capture limit exceeded')
                        captured_bytes += size
                        text.append(fragment)
                        result['chunks'].append(dict(seconds=elapsed, text=fragment))
                        if first_output_event is not None:
                            first_output_event.set()
                usage = item.get('usage')
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise ValueError('invalid benchmark usage object')
                    for key, expected in (('prompt_tokens', len(tokens)), ('completion_tokens', output_tokens)):
                        if type(usage.get(key)) is not int or usage[key] != expected:
                            raise ValueError('benchmark requires exact integer input and output token counts')
                    result['usage'] = usage
        if not done or not text:
            raise ValueError('incomplete or empty generation')
        usage = result.get('usage', {})
        if usage.get('completion_tokens') != output_tokens or usage.get('prompt_tokens') != len(tokens):
            raise ValueError('benchmark requires exact input and output token counts')
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if first_output_event is not None:
            first_output_event.set()
    result['seconds'] = time.perf_counter() - start
    result['text'] = ''.join(text)
    result['captured_utf8_bytes'] = captured_bytes
    result['first_output_seconds'] = result['chunks'][0]['seconds'] if result['chunks'] else None
    # SSE can coalesce tokens; this is explicitly a client-visible proxy, not
    # the engine's internal decode-only throughput or a competitive pass gate.
    if result['passed'] and result['seconds'] > result['first_output_seconds']:
        result['post_first_output_tokens_per_second_proxy'] = (output_tokens - 1) / (
            result['seconds'] - result['first_output_seconds'])
    return result


def prepare_fixed_prompts(tokenizer):
    corpus = tokenizer.encode(''.join(
        f'Record {i}: The parcel is stored in cabinet {i % 11}, drawer {i % 7}.\n'
        for i in range(1000)), add_special_tokens=False).ids
    required = 17 * 3 + 2048
    if not isinstance(corpus, list) or len(corpus) < required:
        raise ValueError('tokenizer corpus is too short for fixed workload prompts')
    if any(type(token) is not int or token < 0 for token in corpus[:required]):
        raise ValueError('tokenizer returned invalid fixed workload token IDs')
    return {str(n): [corpus[17*i:17*i+n] for i in range(4)] for n in (128, 512, 2048)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--engine', choices=('dllm', 'llama.cpp'), required=True)
    p.add_argument('--model', default='a')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--phase', choices=('A1', 'B1', 'B2', 'A2'), required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.path not in ('', '/'):
        p.error('use an isolated loopback HTTP server root')
    base = args.base_url.rstrip('/')
    manifest = json.loads(Path(args.manifest).read_text())
    for key in ('model_revision', 'engine_revision', 'precision', 'hardware'):
        if not manifest.get(key):
            p.error('manifest requires ' + key)
    eos_ids = manifest.get('eos_ids')
    if not isinstance(eos_ids, list) or not eos_ids or any(type(t) is not int or t < 0 for t in eos_ids):
        p.error('manifest requires the checkpoint EOS token IDs')
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompts = prepare_fixed_prompts(tokenizer)
    report = dict(passed=False, competitive_pass=None, manifest=manifest, engine=args.engine,
                  phase=args.phase, context=4096, output_tokens=128, warmup_rounds=3,
                  eos_policy='identical -100 logit bias; fail any request not reaching 128 tokens',
                  measured_rounds=10, tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                  prompts=prompts, rounds=[], summaries=[])
    output = Path(args.output)

    def save():
        save_report(output, report)

    try:
        if args.engine == 'llama.cpp':
            report['context_proof'] = llama_context(base, 4096)
        else:
            with urlopen(base + '/v1/models/status', timeout=30) as response:
                report['context_proof'] = json.load(response)
            if report['context_proof']['context'] != 4096:
                raise ValueError('dllm context capacity must be 4096')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for n, group in prompts.items():
                # Warm actual concurrent groups, including intermediate decode
                # batch sizes that may arise while a four-request group drains.
                for concurrency in (1, 2, 3, 4):
                    count = 13 if concurrency in (1, 4) else 3
                    for trial in range(count):
                        barrier = threading.Barrier(concurrency)
                        start = time.perf_counter()
                        futures = [pool.submit(request, base, args.model, ids, args.engine, eos_ids, barrier)
                                   for ids in group[:concurrency]]
                        results = [f.result() for f in futures]
                        row = dict(prompt_tokens=int(n), concurrency=concurrency, trial=trial,
                                   warmup=trial < 3, seconds=time.perf_counter()-start, requests=results)
                        report['rounds'].append(row)
                        save()
                        if not all(r['passed'] for r in results):
                            raise ValueError('request failed; partial measurements retained')
                        print(f'{args.phase}: input={n} concurrency={concurrency} round={trial+1}/{count}', flush=True)
        for n in prompts:
            for concurrency in (1, 4):
                rows = [r for r in report['rounds'] if not r['warmup'] and
                        r['prompt_tokens'] == int(n) and r['concurrency'] == concurrency]
                requests = [q for r in rows for q in r['requests']]
                report['summaries'].append(dict(prompt_tokens=int(n), concurrency=concurrency,
                    first_output_seconds=distribution([q['first_output_seconds'] for q in requests]),
                    request_seconds=distribution([q['seconds'] for q in requests]),
                    aggregate_output_tokens_per_second=distribution([128*concurrency/r['seconds'] for r in rows]),
                    post_first_output_tokens_per_second_proxy=distribution([
                        q['post_first_output_tokens_per_second_proxy'] for q in requests])))
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
