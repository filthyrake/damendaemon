"""Actual HTTP proof across Qwen2, Qwen3 reasoning modes, and a Llama decoder."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.bench import events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('qwen2')
    parser.add_argument('qwen3')
    parser.add_argument('llama')
    args = parser.parse_args()
    models = {'old': args.qwen2, 'new': args.qwen3, 'llama': args.llama}
    reports = []
    for capacity in (1, 3):
        with tempfile.TemporaryDirectory(prefix='dllm-model-matrix-') as directory:
            logpath = Path(directory) / 'server.log'
            with logpath.open('w') as log:
                command = [sys.executable, '-m', 'dllm', 'serve', '--port', '0', '--context', '2048',
                           '--gpu-models', str(capacity), '--host-models', '3', '--host-gib', '8']
                for alias, path in models.items():
                    command += ['--model', alias+'='+path]
                process = subprocess.Popen(command, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic()+60
                    while time.monotonic() < deadline and process.poll() is None:
                        ready = [line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                        if ready:
                            base = ready[-1].split()[-1]
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError(logpath.read_text())
                    def post(alias, **options):
                        body = {'model': alias, 'messages': [{'role': 'user', 'content':
                            'Explain binary search in detail, using a worked example.'}],
                            'temperature': 0, 'max_tokens': 32}
                        if alias == 'new':
                            body['chat_template_kwargs'] = {'enable_thinking': False}
                        body.update(options)
                        return urlopen(Request(base+'/v1/chat/completions', data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'}), timeout=180)
                    def request(alias, **options):
                        with post(alias, **options) as response:
                            return json.load(response)
                    baselines = {}
                    transitions = []
                    for alias in ('old', 'new', 'llama', 'old', 'new', 'llama'):
                        result = request(alias)
                        text = result['choices'][0]['message']['content']
                        if alias in baselines:
                            assert text == baselines[alias]
                            assert result['dllm_timing']['source_tier'] == ('host' if capacity == 1 else 'gpu')
                        baselines[alias] = text
                        transitions.append({'model':alias, 'usage':result['usage'], 'timing':result['dllm_timing']})
                    # Different models cannot corrupt one another's cache or template controls.
                    with ThreadPoolExecutor(6) as pool:
                        aliases = ('old', 'new', 'llama', 'new', 'old', 'llama')
                        results = list(pool.map(request, aliases))
                    for alias, result in zip(aliases, results):
                        assert result['choices'][0]['message']['content'] == baselines[alias], alias
                    with ThreadPoolExecutor(4) as pool:
                        batched = list(pool.map(request, ['new']*4))
                    assert max(r['dllm_timing']['max_batch_size'] for r in batched) == 4
                    for result in batched:
                        assert result['choices'][0]['message']['content'] == baselines['new']
                    reasoning = request('new', chat_template_kwargs={'enable_thinking':True}, max_tokens=16)
                    assert '<think>' in reasoning['choices'][0]['message']['content']
                    assert '<think>' not in baselines['new']
                    with post('new', stream=True, max_tokens=128) as response:
                        stream = events(response)
                        next(stream)
                        next(stream)
                    assert request('llama')['choices'][0]['message']['content'] == baselines['llama']
                    try:
                        post('llama', chat_template_kwargs={'enable_thinking':False}).close()
                        raise AssertionError('unsupported template option accepted')
                    except HTTPError as error:
                        assert error.code == 400
                        error.close()
                    # A longer, multilingual chat follows the rejection and disconnect.
                    document = '\n'.join(f'Record {i}: cabinet {i%11}.' for i in range(80))
                    long_result = request('new', messages=[{'role':'system','content':'Answer briefly.'},
                        {'role':'user','content':document+'\nこんにちは. Which cabinet stores record 73?'}])
                    assert long_result['usage']['prompt_tokens'] > 512
                    assert long_result['choices'][0]['message']['content']
                    reports.append({'gpu_models':capacity, 'transitions':transitions,
                        'sample_text':baselines, 'concurrent_mixed_models_passed':True,
                        'concurrent_qwen3_passed':True, 'thinking_modes_passed':True,
                        'disconnect_and_rejection_recovery_passed':True,
                        'long_chat':long_result, 'reasoning':reasoning})
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    print(json.dumps({'models':models,'reports':reports,'passed':True},indent=2))


if __name__ == '__main__':
    main()
