"""Sequential Apple GPU HTTP comparison with a separately built llama.cpp server."""
from dllm.evidence_environment import report_environment
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
from dllm.bench import llama_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--gguf', required=True)
    p.add_argument('--llama-server', required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--weight-dtype', choices=('float32', 'float16'), default='float32',
                   help='device weight storage; the supplied GGUF must use the matching type')
    p.add_argument('--engines', nargs='+', choices=('dllm', 'llama.cpp'), default=['dllm', 'llama.cpp'])
    p.add_argument('--concurrencies', nargs='+', type=int, choices=range(1,9), default=[1,2,4])
    p.add_argument('--raw', action='store_true')
    p.add_argument('--llama-override-kv', action='append', default=[], help='explicit GGUF metadata override KEY=TYPE:VALUE')
    p.add_argument('--prompt-token-ids')
    p.add_argument('--max-tokens',type=int,default=64)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DLLM_DEVICE='mps', OMP_NUM_THREADS='4', DLLM_METAL_WEIGHT_DTYPE=args.weight_dtype)
    for engine in args.engines:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        if engine == 'dllm':
            command = [sys.executable, '-m', 'dllm', 'serve', '--device', 'mps', '--context', str(args.context),
                       '--port', str(port), '--batch-size', '4', '--model', 'a='+args.model]
        else:
            command = [args.llama_server, '-m', args.gguf, '--alias', 'a', '--host', '127.0.0.1',
                       '--port', str(port), '-ngl', '999', '-np', '4', '-c', str(args.context*4),
                       '--kv-unified-per-slot', str(args.context), '-ctk', 'f32', '-ctv', 'f32',
                       '--no-cache-prompt', '--cache-ram', '0', '-t', '4',
                       *[part for value in args.llama_override_kv for part in ('--override-kv',value)]]
        (output / f'compare-mps-{engine}-command.json').write_text(json.dumps({
            'command': command, 'device': 'Apple M4 GPU',
            'weights':env.get('DLLM_QUANTIZATION','none') if env.get('DLLM_QUANTIZATION','none')!='none' else args.weight_dtype,
            'unquantized_weight_dtype':args.weight_dtype,'kv': 'float32',
            'runtime_environment':report_environment(env),
            'context_per_slot': args.context, 'cpu_threads': 4, 'dllm_backend': env.get('DLLM_KERNELS', 'auto')}, indent=2))
        with (output / f'compare-mps-{engine}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic()+180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'{engine} exited; inspect its log')
                    try:
                        with urlopen(base+'/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise TimeoutError(f'{engine} startup timed out')
                if engine == 'llama.cpp':
                    (output / f'compare-mps-{engine}-context.json').write_text(json.dumps(llama_context(base, args.context), indent=2))
                for concurrency in args.concurrencies:
                    with (output / f'compare-mps-{engine}-c{concurrency}.json').open('w') as out:
                        subprocess.run([sys.executable, 'scripts/compare_llama.py', base+'/v1',
                            '--engine', engine, '--concurrency', str(concurrency),
                            '--max-tokens',str(args.max_tokens),
                            *(['--raw'] if args.raw else []),
                            *(['--prompt-token-ids',args.prompt_token_ids] if args.prompt_token_ids else [])], env=env,
                            stdout=out, check=True, timeout=300)
                    print(engine, concurrency, 'complete', flush=True)
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
