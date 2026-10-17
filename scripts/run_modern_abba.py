"""Own sequential isolated server lifetimes for the fixed modern HTTP workload.

The result is an engineering comparison. Precision-policy differences and
missing internal decode/memory telemetry explicitly prevent a parity verdict.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--gguf', required=True)
    p.add_argument('--llama-server', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--llama-revision', required=True)
    p.add_argument('--damendaemon-revision', required=True)
    p.add_argument('--hardware', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--candidate-q8-small-batch', action='store_true',
                   help='compare ordinary damendaemon with the isolated HIP Q8 policy instead of llama.cpp')
    args = p.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    model = Path(args.model_path).resolve()
    config = json.loads((model/'config.json').read_text())
    if config['model_type'] != 'qwen3_5':
        raise ValueError('this first comparison targets Qwen3.5')
    try:
        generation = json.loads((model/'generation_config.json').read_text())
    except FileNotFoundError:
        generation = {}
    eos = set()
    for c in (config.get('text_config', config), generation):
        value = c.get('eos_token_id')
        if value is not None:
            eos.update([value] if isinstance(value, int) else value)
    if not eos:
        raise ValueError('missing EOS IDs')
    report = dict(passed=False, competitive_pass=None, phases=[],
                  limitations=[('within-damendaemon isolated Q8 launch policy; identical arithmetic/storage policy' if args.candidate_q8_small_batch else
                                'FP16 damendaemon versus FP32 llama.cpp convolution history and differing intermediate arithmetic'),
                               'client-visible decode proxy; internal decode and peak memory telemetry outstanding'],
                  gguf_sha256=digest(args.gguf), llama_binary_sha256=digest(args.llama_server))
    lock = (directory.parent/'modern-gpu-benchmark.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for phase, engine in (('A1', 'damendaemon'), ('B1', 'llama.cpp'), ('B2', 'llama.cpp'), ('A2', 'damendaemon')):
            candidate = args.candidate_q8_small_batch and phase.startswith('B')
            if candidate:
                engine = 'damendaemon'
            with socket.socket() as reserved:
                reserved.bind(('127.0.0.1', 0))
                port = reserved.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
            manifest = dict(model_revision=args.model_revision, hardware=args.hardware, eos_ids=sorted(eos),
                engine_revision=args.damendaemon_revision if engine == 'damendaemon' else args.llama_revision,
                precision=dict(projections='Q8_0', input_embedding='FP16', output_head='Q8_0',
                    attention_kv='FP16', recurrent_state='FP32',
                    convolution_history='FP16' if engine == 'damendaemon' else 'FP32'),
                graph_policy='normal enabled', prompt_cache=False)
            if args.candidate_q8_small_batch:
                manifest['variant'] = 'hip-q8-small-batch' if candidate else 'ordinary'
                manifest['candidate_source_sha256'] = digest('scripts/experiments/hip_q8_small_batch.py')
            manifest_path = directory/f'{phase}.manifest.json'
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            env = dict(os.environ)
            env.update(ROCR_VISIBLE_DEVICES='0', HIP_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4')
            env.pop('CUDA_VISIBLE_DEVICES', None)
            if engine == 'damendaemon':
                env.update(DAMENDAEMON_DEVICE='rocm', DAMENDAEMON_KERNELS='triton', DAMENDAEMON_QUANTIZATION='q8_0',
                           DAMENDAEMON_ROCM_MEMORY_GIB='8', DAMENDAEMON_WEIGHT_CACHE='')
                command = [sys.executable, '-m', 'damendaemon', 'serve', '--port', str(port),
                           '--context', '4096', '--batch-size', '4', '--gpu-models', '1', '--model', 'a='+str(model)]
                if candidate:
                    command = [sys.executable, 'scripts/experiments/hip_q8_small_batch.py', '--module', 'damendaemon', *command[3:]]
                readiness = base+'/v1/models/status'
            else:
                command = [args.llama_server, '--model', args.gguf, '--alias', 'a', '--host', '127.0.0.1',
                           '--port', str(port), '--ctx-size', '16384', '--parallel', '4',
                           '--n-gpu-layers', '999', '--flash-attn', 'on', '--cache-type-k', 'f16',
                           '--cache-type-v', 'f16', '--threads', '4', '--batch-size', '2048',
                           '--ubatch-size', '512', '--no-context-shift']
                readiness = base+'/health'
            record = dict(phase=phase, engine=engine, command=command, passed=False,
                          readiness_semantics='HTTP ready; damendaemon loads on first request, llama.cpp preloads')
            report['phases'].append(record)
            start = time.monotonic()
            with (directory/f'{phase}.server.log').open('w') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    deadline = start+180
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f'{phase} server exited: see server log')
                        try:
                            with urlopen(readiness, timeout=1) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        time.sleep(.2)
                    else:
                        raise TimeoutError(f'{phase} server readiness timeout')
                    record['process_ready_seconds'] = time.monotonic()-start
                    with (directory/f'{phase}.workload.log').open('w') as workload:
                        subprocess.run([sys.executable, 'scripts/benchmark_modern_http.py', '--base-url', base,
                            '--engine', engine, '--model', 'a', '--tokenizer', str(model/'tokenizer.json'),
                            '--manifest', str(manifest_path), '--phase', phase,
                            '--output', str(directory/f'{phase}.json')], env=env, stdout=workload,
                            stderr=subprocess.STDOUT, check=True, timeout=1800)
                    record['passed'] = True
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            record['phase_seconds'] = time.monotonic()-start
            (directory/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        (directory/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        lock.close()


if __name__ == '__main__':
    main()
