"""Actual CUDA GGUF proof against a value-identical HF checkpoint preparation.

The comparator GGUF must have been exported with make_q8_comparator.py. This
checks import/execution identity, not the quality of arbitrary quantizations.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gc
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import torch
from dllm.bench import events
from dllm.device import empty_cache, device_name
from dllm.gguf_model import GgufWeights
from dllm.qwen2 import ModelWeights
from dllm.runtime import make_executor
from dllm.prompts import decode_output


def decoded_trace(model, tokens):
    stop = next((i for i, token in enumerate(tokens) if token in model.eos), None)
    visible = tokens if stop is None else tokens[:stop+1]
    return dict(text=decode_output(model, visible), completion_tokens=len(tokens) if stop is None else stop+1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf')
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--allow-tokenizer-differences', action='store_true',
                   help='record source tokenizer differences; require exact source-decoded token traces instead of cross-source text identity')
    args = p.parse_args()
    if args.context < 512:p.error('--context must be at least 512')
    torch.set_num_threads(4)
    source = GgufWeights(args.gguf, args.context, 'none')
    reference = ModelWeights(args.hf, args.context, source.quantization)
    chat_messages = [dict(role='user', content='Explain binary search with a short worked example.')]
    chat_inputs = dict(gguf=source.chat_tokens(chat_messages), hf=reference.chat_tokens(chat_messages))
    if chat_inputs['gguf'] != chat_inputs['hf'] and not args.allow_tokenizer_differences:
        raise ValueError('chat inputs differ; diagnose source tokenizer semantics before allowing differences')
    raw_ids = source.raw_tokens('Explain binary search step by step:')
    window = source.config.get('sliding_window')
    boundary_prompts = []
    if window and args.context >= window + 8:
        corpus = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
        boundary_prompts = [(corpus * ((length + len(corpus) - 1)//len(corpus)))[:length]
                            for length in (window-1, window, window+1) if length > 0]
    for name, value in reference.weights.items():
        assert torch.equal(value, source.weights[name]), name
    for name, value in reference.scales.items():
        assert torch.equal(value, source.scales[name]), name
    report = dict(gguf=str(Path(args.gguf).resolve()), hf=str(Path(args.hf).resolve()), device=device_name(),
                  torch=torch.__version__, quantization=source.quantization,
                  weights_exact=len(reference.weights), scales_exact=len(reference.scales),
                  context=args.context, sliding_window=window, chat_input_ids=chat_inputs,
                  allow_tokenizer_differences=args.allow_tokenizer_differences, raw_prompt_ids=raw_ids, cases=[])
    quantization = source.quantization
    cases = [(f'raw-{length}', ([9707, 11, 1879, 0] * ((length+3)//4))[:length], 8)
             for length in (1, 17, 129, args.context-7)]
    cases += [('chat-gguf', chat_inputs['gguf'], 32), ('chat-hf', chat_inputs['hf'], 32),
              ('warm-raw', raw_ids, 16)]
    cases += [(f'boundary-{len(ids)}', ids, min(8, args.context-len(ids))) for ids in boundary_prompts]
    expected_http = {}
    with torch.inference_mode():
        for graph in (False, True):
            a = make_executor(reference, use_graph=graph)
            b = make_executor(source, use_graph=graph)
            for label, ids, steps in cases:
                length = len(ids)
                expected, actual = a.prefill(ids), b.prefill(ids)
                output = []; errors = []; exact = 0
                for _ in range(steps):
                    equal = torch.equal(expected, actual); exact += int(equal)
                    error = actual.float() - expected.float()
                    absolute = error.abs().max().item()
                    relative = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                    errors.append(dict(max_absolute=absolute, relative_rms=relative))
                    if 'rope_factors' in source.weights:
                        # Llama GGUF stores explicit FP32 divisors. Their rounded
                        # inverse schedule differs slightly from HF's formula.
                        assert absolute < .5 and relative < .01 and expected.argmax(-1).item()==actual.argmax(-1).item(), (graph,length,errors[-1])
                    else:
                        assert equal, (graph, length, absolute)
                    token = expected.argmax(-1).item(); output.append(token)
                    if len(output) < steps:
                        expected, _ = a.decode(token); actual, _ = b.decode(token)
                decoded = dict(gguf=decoded_trace(source, output), hf=decoded_trace(reference, output))
                if graph:assert decoded == expected_http[label]
                else:expected_http[label] = decoded
                report['cases'].append(dict(graph=graph, case=label, prompt_tokens=length, prompt_ids=ids,
                                           qualified_steps=steps, exact_logit_steps=exact, errors=errors, output_tokens=output))
            del a, b, expected, actual
            gc.collect(); empty_cache()
    del source, reference
    report['http_expected'] = expected_http
    gc.collect(); empty_cache()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'server.log'
        with path.open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'dllm', 'serve', '--port', '0',
                '--context', str(args.context), '--gpu-models', '1', '--host-models', '2', '--batch-size', '2'], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and process.poll() is None:
                    ready = [line for line in path.read_text().splitlines() if line.startswith('dllm listening on ')]
                    if ready: base = ready[-1].split()[-1]; break
                    time.sleep(.1)
                else: raise RuntimeError(path.read_text())
                def post(route, body):
                    try:
                        return urlopen(Request(base+route, data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'}), timeout=180)
                    except HTTPError as error:
                        raise RuntimeError(error.read().decode()) from error
                def request(route, body):
                    with post(route, body) as response: return json.load(response)
                def check_trace(response, label, alias, chat=False):
                    text = response['choices'][0]['message']['content'] if chat else response['choices'][0]['text']
                    assert text == expected_http[label][alias]['text'], (label, alias, text, expected_http[label][alias])
                    assert response['usage']['completion_tokens'] == expected_http[label][alias]['completion_tokens']
                for alias, model, policy in (('gguf', args.gguf, 'none'), ('hf', args.hf, quantization)):
                    request('/v1/models/register', dict(model=alias, path=str(Path(model).resolve()), quantization=policy))
                def chat(alias='gguf', **options):
                    return dict(model=alias, messages=chat_messages, temperature=0, max_tokens=32, **options)
                baseline = request('/v1/chat/completions', chat())
                check_trace(baseline, 'chat-gguf', 'gguf', chat=True)
                expected = baseline['choices'][0]['message']['content']
                other = request('/v1/chat/completions', chat('hf'))
                hf_chat = other['choices'][0]['message']['content']
                check_trace(other, 'chat-hf', 'hf', chat=True)
                if not args.allow_tokenizer_differences:
                    assert hf_chat == expected
                matched = request('/v1/completions', dict(model='hf', prompt=chat_inputs['gguf'],
                                  temperature=0, max_tokens=32))
                check_trace(matched, 'chat-gguf', 'hf')
                assert matched['usage'] == baseline['usage']
                restored = request('/v1/chat/completions', chat())
                assert restored['choices'][0]['message']['content'] == expected
                assert restored['dllm_timing']['source_tier'] == 'host'
                with post('/v1/chat/completions', chat(stream=True)) as response:
                    chunks = list(events(response, require_terminated=True))
                assert chunks[-1] == '[DONE]'
                text = ''.join(c.get('delta', {}).get('content', '') for item in chunks[:-1] for c in json.loads(item)['choices'])
                assert text == expected
                with ThreadPoolExecutor(2) as pool:
                    concurrent = list(pool.map(lambda _: request('/v1/chat/completions', chat()), range(2)))
                assert all(r['choices'][0]['message']['content'] == expected for r in concurrent)
                with post('/v1/chat/completions', chat(stream=True)) as response:
                    stream = events(response, require_terminated=True); next(stream); next(stream)
                assert request('/v1/chat/completions', chat())['choices'][0]['message']['content'] == expected
                boundary_checks = []
                for ids in boundary_prompts:
                    body = dict(prompt=ids, temperature=0, max_tokens=min(8,args.context-len(ids)))
                    first = request('/v1/completions', dict(model='gguf', **body))
                    other = request('/v1/completions', dict(model='hf', **body))
                    target = first['choices'][0]['text']
                    check_trace(first, f'boundary-{len(ids)}', 'gguf')
                    check_trace(other, f'boundary-{len(ids)}', 'hf')
                    if not args.allow_tokenizer_differences:
                        assert target == other['choices'][0]['text']
                    assert first['usage'] == other['usage'] and first['usage']['completion_tokens'] > 0
                    with post('/v1/completions', dict(model='gguf', stream=True, **body)) as response:
                        chunks = list(events(response, require_terminated=True))
                    assert chunks[-1] == '[DONE]'
                    streamed = ''.join(c.get('text', '') for item in chunks[:-1] for c in json.loads(item)['choices'])
                    assert streamed == target
                    boundary_checks.append(dict(prompt_tokens=len(ids), usage=first['usage'], text=target,
                                                hf_text=other['choices'][0]['text'],
                                                hf_gguf_text_equal=target == other['choices'][0]['text'],
                                                exact_decoded_traces=True, stream_equal=True))
                timing = []
                for turn in range(4):
                    for alias in (('gguf', 'hf') if turn % 2 == 0 else ('hf', 'gguf')):
                        body = dict(model=alias, prompt=raw_ids, temperature=0, max_tokens=16)
                        request('/v1/completions', body)  # model restore and graph warmup excluded
                        start = time.perf_counter(); result = request('/v1/completions', body)
                        elapsed = time.perf_counter()-start
                        check_trace(result, 'warm-raw', alias)
                        timing.append(dict(model=alias, seconds=elapsed, usage=result['usage'], text=result['choices'][0]['text']))
                if args.allow_tokenizer_differences:
                    assert len({json.dumps(r['usage'],sort_keys=True) for r in timing}) == 1
                else:
                    assert len({r['text'] for r in timing}) == 1
                for alias in ('gguf', 'hf'):
                    status = request('/v1/models/unload', dict(model=alias, release_host=True))
                assert status['host_cached_bytes'] == 0
                assert all(m['residency'] == 'disk' for m in status['models'])
                reloaded = request('/v1/chat/completions', chat())
                assert reloaded['choices'][0]['message']['content'] == expected
                assert reloaded['dllm_timing']['source_tier'] == 'disk'
                report['http'] = dict(registration=True, exact_chat=hf_chat == expected and chat_inputs['gguf'] == chat_inputs['hf'],
                    hf_chat_sample=hf_chat, exact_decoded_traces=True, stream=True, concurrency=True,
                    cancellation_recovery=True, host_restore=True, unload=True, disk_reload=True,
                    sample=expected, warm_raw_trials=timing,
                    window_boundaries=boundary_checks,
                    warm_medians={alias:statistics.median(r['seconds'] for r in timing if r['model'] == alias) for alias in ('gguf', 'hf')})
            except Exception:
                print(path.read_text()[-8000:], file=sys.stderr); raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
    report['passed'] = True
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
