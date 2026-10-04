"""Sequential, isolated A2 HTTP comparison; vLLM is a benchmark-only dependency.

Run from the lab workspace containing model-a.json and baselines/vllm-venv.
Never starts both engines at once. Only terminates process groups it creates.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-metadata', default='model-a.json')
    parser.add_argument('--label', default='batched')
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.5)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--engines', nargs='+', choices=('dllm', 'vllm'), default=['dllm', 'vllm'])
    parser.add_argument('--thinking', choices=('auto', 'on', 'off'), default='auto')
    args = parser.parse_args()
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('label must contain only letters, digits, hyphens or underscores')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU memory utilization must be between zero and one')
    root = Path.cwd()
    uuid = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('this benchmark is restricted to the authorized A2 UUID')
    model = json.loads((root / args.model_metadata).read_text())
    for engine in args.engines:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ)
        if engine == 'dllm':
            command = [sys.executable, '-m', 'dllm', 'serve', '--port', str(port), '--context', str(args.context),
                       '--batch-size', '4', '--model', 'a=' + model['path']]
        else:
            venv = root / 'baselines/vllm-venv/bin'
            command = [str(venv / 'vllm'), 'serve', model['path'], '--served-model-name', 'a',
                       '--host', '127.0.0.1', '--port', str(port), '--dtype', 'half',
                       '--max-model-len', str(args.context), '--max-num-seqs', '4', '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                       '--no-enable-prefix-caching', '--generation-config', 'vllm']
            config = json.loads((Path(model['path']) / 'config.json').read_text())
            if config.get('vision_config') is not None:
                command += ['--language-model-only']
            env.update(PATH=str(venv) + ':' + env.get('PATH', ''), VLLM_NO_USAGE_STATS='1',
                       DO_NOT_TRACK='1', HF_HUB_OFFLINE='1',
                       VLLM_CACHE_ROOT=str(root / 'baselines/vllm-cache'),
                       TORCHINDUCTOR_CACHE_DIR=str(root / 'baselines/inductor-cache'),
                       TORCHINDUCTOR_COMPILE_THREADS='2', MAX_JOBS='2')
        (root / f'compare-{engine}-{args.label}-command.json').write_text(json.dumps({
            'command': command, 'model': model, 'gpu_uuid': uuid,
            'runtime_environment': {key:value for key,value in env.items() if key.startswith('DLLM_')}}, indent=2))
        with (root / f'compare-{engine}-{args.label}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'{engine} exited; inspect its benchmark log')
                    try:
                        with urlopen(base + '/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(.5)
                else:
                    raise TimeoutError(f'{engine} startup timed out')
                for concurrency in (1, 2, 4):
                    name = f'compare-{engine}-{args.label}-c{concurrency}.json'
                    with (root / name).open('w') as out:
                        subprocess.run([sys.executable, 'scripts/compare_llama.py', base + '/v1',
                                        '--engine', engine, '--concurrency', str(concurrency), '--thinking', args.thinking],
                                       stdout=out, check=True, timeout=180)
                    print(name, 'complete', flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


if __name__ == '__main__':
    main()
