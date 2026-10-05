"""Actual AWQ factory, allocation release and HTTP lifecycle qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
import weakref

import torch
from dllm.bench import events
from dllm.models import load
from dllm.prompts import decode_output
from dllm.runtime import make_executor, require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experimental-awq-decode', action='store_true')
    args = parser.parse_args()
    require_gpu()
    experiment = None
    if args.experimental_awq_decode:
        from experiments.awq_decode import install_validation_dispatch
        experiment = install_validation_dispatch()
    torch.set_num_threads(4)
    path = Path(args.model).resolve()
    source = load(path, args.context, 'none')
    if source.weight_format != 'awq_gemm':
        raise ValueError('this gate requires native AWQ')
    messages = [dict(role='user', content='Explain binary search with a short worked example.')]
    prompt = source.chat_tokens(messages)
    report = dict(passed=False, experiment=experiment, model=str(path), device=torch.cuda.get_device_name(),
                  context=args.context, host_bytes=source.bytes, cycles=[],
                  alias_scope='two aliases of the same pinned AWQ checkpoint',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    previous = None
    try:
        with torch.inference_mode():
            for cycle in range(3):
                executor = make_executor(source, use_graph=True, max_batch=4)
                refs = [weakref.ref(executor)]
                tensors = [*executor.w.values(), *executor.quant_linear.scales.values(),
                           *(m.zeros for m in executor.quant_linear.awq.values())]
                refs += [weakref.ref(t) for t in tensors]
                del tensors
                logits = executor.prefill_batch([prompt]*4, list(range(4)))
                logits, tokens = executor.decode_batch(logits.argmax(-1).tolist(), list(range(4)))
                if not torch.isfinite(logits).all():
                    raise AssertionError('nonfinite batch logits')
                logits = executor.prefill(prompt)
                output = []
                for _ in range(32):
                    token = logits.argmax(-1).item()
                    output.append(token)
                    if token in source.eos:
                        break
                    logits, next_tokens = executor.decode(token)
                    del next_tokens
                text = decode_output(source, output)
                if cycle == 0:
                    expected = text
                    report['expected'] = dict(text=text, tokens=output)
                elif text != expected:
                    raise AssertionError('generation changed after executor recreation')
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                del executor, logits, tokens
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                released = torch.cuda.memory_allocated()
                item = dict(resident=resident, after_release=released,
                            references_released=all(ref() is None for ref in refs))
                report['cycles'].append(item)
                if not item['references_released']:
                    raise AssertionError('AWQ executor retains GPU weights/scales/zeros')
                if resident-released < source.bytes:
                    raise AssertionError('packed model allocation was not released')
                if previous is not None and released > previous+1024**2:
                    raise AssertionError('repeated unload grows retained allocation')
                previous = released
        del source
        gc.collect()
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='dllm-awq-http-') as folder:
            log_path = Path(folder)/'server.log'
            with log_path.open('w') as log:
                module = 'scripts.experiments.serve_awq_decode' if args.experimental_awq_decode else 'dllm'
                process = subprocess.Popen([sys.executable, '-m', module, 'serve', '--port', '0',
                    '--context', str(args.context), '--gpu-models', '1', '--host-models', '2',
                    '--batch-size', '2'], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic()+120
                    while time.monotonic() < deadline and process.poll() is None:
                        ready = [line for line in log_path.read_text().splitlines() if line.startswith('dllm listening on ')]
                        if ready:
                            base = ready[-1].split()[-1]
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError('HTTP server did not become ready')

                    def post(route, body):
                        return urlopen(Request(base+route, data=json.dumps(body).encode(),
                            headers={'Content-Type': 'application/json'}), timeout=180)

                    def request(route, body):
                        with post(route, body) as response:
                            return json.load(response)

                    def chat(alias='awq', **options):
                        return dict(model=alias, messages=messages, temperature=0, max_tokens=32, **options)

                    def check(response):
                        if response['choices'][0]['message']['content'] != expected:
                            raise AssertionError('HTTP generation differs from direct packed executor')
                        if response['usage']['completion_tokens'] != len(report['expected']['tokens']):
                            raise AssertionError('HTTP token accounting differs')

                    for alias in ('awq', 'alternate'):
                        request('/v1/models/register', dict(model=alias, path=str(path), quantization='none'))
                    first = request('/v1/chat/completions', chat())
                    check(first)
                    check(request('/v1/chat/completions', chat('alternate')))
                    restored = request('/v1/chat/completions', chat())
                    check(restored)
                    if restored['dllm_timing']['source_tier'] != 'host':
                        raise AssertionError('alias switching did not restore host weights')
                    with post('/v1/chat/completions', chat(stream=True)) as response:
                        chunks = list(events(response))
                    streamed = ''.join(c.get('delta', {}).get('content', '') for item in chunks[:-1]
                                       for c in json.loads(item)['choices'])
                    if chunks[-1] != '[DONE]' or streamed != expected:
                        raise AssertionError('stream differs from nonstreaming output')
                    with ThreadPoolExecutor(2) as pool:
                        for response in pool.map(lambda _: request('/v1/chat/completions', chat()), range(2)):
                            check(response)
                    with post('/v1/chat/completions', chat(stream=True)) as response:
                        stream = events(response)
                        next(stream)
                        next(stream)
                    check(request('/v1/chat/completions', chat()))
                    for alias in ('awq', 'alternate'):
                        status = request('/v1/models/unload', dict(model=alias, release_host=True))
                    if status['host_cached_bytes'] or any(m['residency'] != 'disk' for m in status['models']):
                        raise AssertionError('unload retained model residency')
                    reloaded = request('/v1/chat/completions', chat())
                    check(reloaded)
                    if reloaded['dllm_timing']['source_tier'] != 'disk':
                        raise AssertionError('unloaded model did not reload from disk')
                    report['http'] = dict(registration=True, exact_direct_trace=True, stream=True,
                        concurrency=True, cancellation_recovery=True, host_restore=True,
                        unload=True, disk_reload=True)
                except Exception:
                    report['server_log_tail'] = log_path.read_text()[-8000:]
                    raise
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with (path/'model.safetensors').open('rb') as stream:
            report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
