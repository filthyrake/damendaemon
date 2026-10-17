"""Sequential, isolated A2 HTTP comparison; vLLM is a benchmark-only dependency.

Run from the lab workspace containing model-a.json and baselines/vllm-venv.
Never starts both engines at once. Only terminates process groups it creates.
"""
from damendaemon.evidence_environment import report_environment
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from damendaemon.bench import llama_context
from damendaemon.token_workload import load_token_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-metadata', default='model-a.json')
    parser.add_argument('--label', default='batched')
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.5)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--engines', nargs='+', choices=('damendaemon', 'vllm', 'llama.cpp'), default=['damendaemon', 'vllm'])
    parser.add_argument('--gguf', help='matching checkpoint and precision for llama.cpp')
    parser.add_argument('--llama-kv-dtype', choices=('f16','f32'), default='f16', help='match the selected runtime KV precision')
    parser.add_argument('--thinking', choices=('auto', 'on', 'off'), default='auto')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--llama-override-kv', action='append', default=[], help='explicit GGUF metadata override KEY=TYPE:VALUE')
    parser.add_argument('--prompt-token-ids')
    parser.add_argument('--top-logprobs',type=int,choices=range(21),default=None)
    parser.add_argument('--max-tokens',type=int,default=64)
    parser.add_argument('--llama-flash-attn',choices=('on','off','auto'),default='on')
    parser.add_argument('--batch-size',type=int,choices=range(1,9),default=4)
    parser.add_argument('--concurrencies',nargs='+',type=int,choices=range(1,9),default=[1,2,4])
    parser.add_argument('--host-gib', type=float, default=8)
    parser.add_argument('--dtype', choices=('half','float'), default='half', help='vLLM model arithmetic/storage precision')
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument('--experimental-awq-decode', action='store_true')
    experiments.add_argument('--experimental-awq-marlin', metavar='LIBRARY')
    experiments.add_argument('--experimental-olmoe-baseline', metavar='SOURCE')
    experiments.add_argument('--experimental-olmoe-prefill-baseline', metavar='SOURCE')
    args = parser.parse_args()
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('label must contain only letters, digits, hyphens or underscores')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU memory utilization must be between zero and one')
    if 'llama.cpp' in args.engines and not args.gguf:
        parser.error('llama.cpp requires --gguf with matching weights and precision')
    if not 1 <= args.context <= 2**20:
        parser.error('context must be 1..1048576 tokens')
    if not 2 <= args.max_tokens < args.context:
        parser.error('max tokens must be at least two and smaller than context')
    if not 1 <= args.startup_timeout <= 86400:
        parser.error('startup timeout must be 1..86400 seconds')
    host_bytes = args.host_gib * 1024**3
    if not math.isfinite(host_bytes) or host_bytes < 1:
        parser.error('host memory budget must be finite and at least one byte')
    if (args.raw or args.prompt_token_ids) and args.top_logprobs is not None:
        parser.error('logprob benchmark currently requires chat completion')
    if args.prompt_token_ids:
        try:load_token_prompt(args.prompt_token_ids,max_tokens=args.context-args.max_tokens)
        except (OSError,ValueError):parser.error('invalid or over-context prompt token file')
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
        if engine == 'damendaemon':
            module = 'scripts.experiments.serve_awq_decode' if args.experimental_awq_decode else 'damendaemon'
            if args.experimental_awq_marlin:
                module = 'scripts.experiments.serve_awq_marlin'
                env['DAMENDAEMON_EXPERIMENT_MARLIN_LIBRARY'] = str(Path(args.experimental_awq_marlin).resolve())
            moe_baseline = args.experimental_olmoe_baseline or args.experimental_olmoe_prefill_baseline
            if moe_baseline:
                module = 'scripts.experiments.serve_moe_baseline'
                env['DAMENDAEMON_EXPERIMENT_MOE_BASELINE'] = str(Path(moe_baseline).resolve())
                env['DAMENDAEMON_EXPERIMENT_MOE_METHOD'] = ('prefill_batch' if args.experimental_olmoe_prefill_baseline else 'experts')
                env['DAMENDAEMON_EXPERIMENT_MOE_REPORT'] = str(root / f'compare-{engine}-{args.label}-source.json')
            command = [sys.executable, '-m', module, 'serve', '--port', str(port), '--context', str(args.context),
                       '--batch-size', str(args.batch_size), '--host-gib', str(args.host_gib), '--model', 'a=' + model['path']]
        elif engine == 'llama.cpp':
            command = [str(root / 'baselines/llama-cuda-build/bin/llama-server'),
                       '-m', args.gguf, '--alias', 'a', '--host', '127.0.0.1', '--port', str(port),
                       '-ngl', '99', '-c', str(args.context*args.batch_size), '-np', str(args.batch_size),
                       '--kv-unified-per-slot', str(args.context),
                       '--cache-type-k', args.llama_kv_dtype, '--cache-type-v', args.llama_kv_dtype,
                       '--flash-attn', args.llama_flash_attn, '--cache-ram', '0', '--jinja',
                       *[part for value in args.llama_override_kv for part in ('--override-kv',value)]]
        else:
            venv = root / 'baselines/vllm-venv/bin'
            command = [str(venv / 'vllm'), 'serve', model['path'], '--served-model-name', 'a',
                       '--host', '127.0.0.1', '--port', str(port), '--dtype', args.dtype,
                       '--max-model-len', str(args.context), '--max-num-seqs', str(args.batch_size), '--gpu-memory-utilization', str(args.gpu_memory_utilization),
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
            'runtime_environment': report_environment(env)}, indent=2))
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
                if engine == 'llama.cpp':
                    (root / f'compare-{engine}-{args.label}-context.json').write_text(json.dumps(llama_context(base, args.context), indent=2))
                for concurrency in args.concurrencies:
                    name = f'compare-{engine}-{args.label}-c{concurrency}.json'
                    with (root / name).open('w') as out:
                        subprocess.run([sys.executable, 'scripts/compare_llama.py', base + '/v1',
                                        '--engine', engine, '--concurrency', str(concurrency), '--thinking', args.thinking,
                                        '--max-tokens',str(args.max_tokens),
                                        *(['--raw'] if args.raw else []),
                                        *(['--top-logprobs',str(args.top_logprobs)] if args.top_logprobs is not None else []),
                                        *(['--prompt-token-ids', args.prompt_token_ids] if args.prompt_token_ids else [])],
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
