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


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf'); args = p.parse_args()
    torch.set_num_threads(4)
    source = GgufWeights(args.gguf, 512, 'none')
    reference = ModelWeights(args.hf, 512, source.quantization)
    for name, value in reference.weights.items():
        assert torch.equal(value, source.weights[name]), name
    for name, value in reference.scales.items():
        assert torch.equal(value, source.scales[name]), name
    report = dict(gguf=str(Path(args.gguf).resolve()), hf=str(Path(args.hf).resolve()), device=device_name(),
                  torch=torch.__version__, quantization=source.quantization,
                  weights_exact=len(reference.weights), scales_exact=len(reference.scales), cases=[])
    quantization = source.quantization
    with torch.inference_mode():
        for graph in (False, True):
            a = make_executor(reference, use_graph=graph)
            b = make_executor(source, use_graph=graph)
            for length in (1, 17, 129, 505):
                ids = ([9707, 11, 1879, 0] * 127)[:length]
                expected, actual = a.prefill(ids), b.prefill(ids)
                output = []
                for _ in range(8):
                    assert torch.equal(expected, actual), (graph, length, float((expected - actual).abs().max()))
                    token = expected.argmax(-1).item(); output.append(token)
                    if len(output) < 8:
                        expected, _ = a.decode(token); actual, _ = b.decode(token)
                report['cases'].append(dict(graph=graph, prompt_tokens=length, exact_logit_steps=8, output_tokens=output))
            del a, b, expected, actual
            gc.collect(); empty_cache()
    del source, reference
    gc.collect(); empty_cache()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'server.log'
        with path.open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'dllm', 'serve', '--port', '0',
                '--context', '512', '--gpu-models', '1', '--host-models', '2', '--batch-size', '2'], stdout=log, stderr=log)
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
                for alias, model, policy in (('gguf', args.gguf, 'none'), ('hf', args.hf, quantization)):
                    request('/v1/models/register', dict(model=alias, path=str(Path(model).resolve()), quantization=policy))
                prompt = 'Explain binary search with a short worked example.'
                def chat(alias='gguf', **options):
                    return dict(model=alias, messages=[{'role':'user','content':prompt}], temperature=0, max_tokens=32, **options)
                baseline = request('/v1/chat/completions', chat())
                expected = baseline['choices'][0]['message']['content']
                other = request('/v1/chat/completions', chat('hf'))
                assert other['choices'][0]['message']['content'] == expected
                restored = request('/v1/chat/completions', chat())
                assert restored['choices'][0]['message']['content'] == expected
                assert restored['dllm_timing']['source_tier'] == 'host'
                with post('/v1/chat/completions', chat(stream=True)) as response:
                    chunks = list(events(response))
                assert chunks[-1] == '[DONE]'
                text = ''.join(c.get('delta', {}).get('content', '') for item in chunks[:-1] for c in json.loads(item)['choices'])
                assert text == expected
                with ThreadPoolExecutor(2) as pool:
                    concurrent = list(pool.map(lambda _: request('/v1/chat/completions', chat()), range(2)))
                assert all(r['choices'][0]['message']['content'] == expected for r in concurrent)
                with post('/v1/chat/completions', chat(stream=True)) as response:
                    stream = events(response); next(stream); next(stream)
                assert request('/v1/chat/completions', chat())['choices'][0]['message']['content'] == expected
                timing = []
                for turn in range(4):
                    for alias in (('gguf', 'hf') if turn % 2 == 0 else ('hf', 'gguf')):
                        body = dict(model=alias, prompt='Explain binary search step by step:', temperature=0, max_tokens=16)
                        request('/v1/completions', body)  # model restore and graph warmup excluded
                        start = time.perf_counter(); result = request('/v1/completions', body)
                        timing.append(dict(model=alias, seconds=time.perf_counter()-start, usage=result['usage'], text=result['choices'][0]['text']))
                assert len({r['text'] for r in timing}) == 1
                for alias in ('gguf', 'hf'):
                    status = request('/v1/models/unload', dict(model=alias, release_host=True))
                assert status['host_cached_bytes'] == 0
                assert all(m['residency'] == 'disk' for m in status['models'])
                report['http'] = dict(registration=True, exact_chat=True, stream=True, concurrency=True,
                    cancellation_recovery=True, host_restore=True, unload=True, sample=expected, warm_raw_trials=timing,
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
