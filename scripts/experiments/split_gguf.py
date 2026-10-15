"""Bounded single-file versus split GGUF correctness qualification (not timing)."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def digest(file, count):
    result = hashlib.sha256()
    while count:
        block = file.read(min(count, 1024*1024))
        if not block:
            raise ValueError('truncated checkpoint')
        result.update(block)
        count -= len(block)
    return result.hexdigest()


def snapshot(path):
    from dllm.gguf_model import plan, identity
    source = plan(path)
    if source.config['model_type'] not in ('qwen2', 'qwen3', 'llama'):
        raise ValueError('gate supports only small dense Qwen2/Qwen3/Llama')
    if source.host_bytes > 1024**3:
        raise ValueError('gate limits host weights to 1 GiB')
    if sum(part.stat().st_size for part, _ in source.identity) > 2*1024**3:
        raise ValueError('gate limits checkpoint files to 2 GiB total')
    files = []
    for part, stamp in source.identity:
        with part.open('rb') as file:
            files.append(dict(path=str(part), sha256=digest(file, part.stat().st_size)))
    tensors = {}
    for name, tensor in sorted(source.tensors.items()):
        with source.identity[tensor.file_index][0].open('rb') as file:
            file.seek(tensor.offset)
            tensors[name] = dict(shape=list(tensor.shape), kind=tensor.kind,
                                 sha256=digest(file, tensor.length))
    if any(identity(part) != stamp for part, stamp in source.identity):
        raise ValueError('checkpoint changed during preparation')
    semantics = dict(config=source.config, tokenizer=source.tokenizer_json,
                     template=source.template, special_tokens=source.special_tokens,
                     eos=sorted(source.eos), quantization=source.quantization,
                     sentencepiece_prefix=source.sentencepiece_prefix, tensors=tensors)
    return dict(files=files, semantics=semantics)


def prepare(single, split, output, context=128, steps=4):
    if not 32 <= context <= 128 or not 1 <= steps <= 8:
        raise ValueError('context must be 32..128 and steps 1..8')
    a, b = snapshot(single), snapshot(split)
    if len(a['files']) != 1 or len(b['files']) < 2:
        raise ValueError('provide a single checkpoint and the first member of a split set')
    if a['semantics'] != b['semantics']:
        raise ValueError('single and split tensor bytes or metadata differ')
    if context > a['semantics']['config']['max_position_embeddings']:
        raise ValueError('context exceeds model position limit')
    # Resolve token IDs on CPU; every subsequent phase uses exactly these IDs.
    from dllm.gguf_model import GgufWeights
    model = GgufWeights(single, context, 'none')
    ids = model.raw_tokens('Hello')
    if not ids or len(ids)+steps > context:
        raise ValueError('prompt does not fit bounded context')
    output.mkdir()  # Never overwrite previous evidence.
    manifest = dict(schema=1, context=context, steps=steps, prompt_ids=ids,
                    allocator_mib=1536, timeout_seconds=120,
                    sources=dict(single=a, split=b), hardware_validated=False)
    save(output/'manifest.json', manifest)
    return manifest


def verify_files(source):
    for item in source['files']:
        path = Path(item['path'])
        with path.open('rb') as file:
            if digest(file, path.stat().st_size) != item['sha256']:
                raise ValueError('checkpoint changed after preparation')


def trace(model, ids, steps):
    import torch
    from dllm.runtime import make_executor
    from dllm.prompts import decode_output
    executor = make_executor(model, use_graph=False)
    tokens, hashes = [], []
    try:
        with torch.inference_mode():
            logits = executor.prefill(ids)
            for index in range(steps):
                cpu = logits.detach().float().cpu().contiguous()
                if not torch.isfinite(cpu).all():
                    raise ValueError('nonfinite logits')
                hashes.append(hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
                token = int(cpu.argmax(-1).item())
                tokens.append(token)
                if token in model.eos:
                    break
                if index+1 < steps:
                    logits, _ = executor.decode(token)
        return dict(tokens=tokens, logits_sha256=hashes,
                    text=decode_output(model, tokens), completion_tokens=len(tokens))
    finally:
        if hasattr(executor, 'close'):
            executor.close()


def http_check(runtime, ids, steps, expected):
    from dllm.server import BoundedHTTPServer, handler_for
    from dllm.bench import events
    server = BoundedHTTPServer(('127.0.0.1', 0), handler_for(runtime), max_connections=2)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05})
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def post(route, body):
        return urlopen(Request(base+route, json.dumps(body).encode(),
                       {'Content-Type': 'application/json'}), timeout=10)

    def request(route, body):
        with post(route, body) as response:
            return json.load(response)

    body = dict(model='a', prompt=ids, max_tokens=steps, temperature=0)
    try:
        first = request('/v1/completions', body)
        if (first['choices'][0]['text'] != expected['text'] or
                first['usage']['completion_tokens'] != expected['completion_tokens']):
            raise ValueError('HTTP output differs from direct numerical trace')
        with post('/v1/completions', dict(body, stream=True)) as response:
            chunks = list(events(response))
        if not chunks or chunks[-1] != '[DONE]':
            raise ValueError('incomplete HTTP stream')
        streamed = ''.join(c.get('text', '') for chunk in chunks[:-1]
                           for c in json.loads(chunk)['choices'])
        if streamed != expected['text']:
            raise ValueError('HTTP stream mismatch')
        status = request('/v1/models/unload', dict(model='a', release_host=True))
        if status['host_cached_bytes'] != 0 or any(m['residency'] != 'disk' for m in status['models']):
            raise ValueError('unload did not release residency')
        restored = request('/v1/completions', body)
        if (restored['choices'][0]['text'] != expected['text'] or
                restored['usage'] != first['usage'] or restored['dllm_timing']['source_tier'] != 'disk'):
            raise ValueError('disk reload mismatch')
        return dict(text=streamed, usage=first['usage'], stream=True, unload=True, disk_reload=True)
    finally:
        server.shutdown()
        try:
            runtime.close()
        finally:
            server.server_close()
            thread.join()


def worker(folder, alias, phase):
    import torch
    manifest = json.loads((folder/'manifest.json').read_text())
    source = manifest['sources'][alias]
    verify_files(source)
    if not torch.cuda.is_available() or torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('requires exactly one visible NVIDIA GPU')
    torch.set_num_threads(1)
    cap = manifest['allocator_mib']*1024**2
    torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats()
    path = source['files'][0]['path']
    if phase == 'trace':
        from dllm.gguf_model import GgufWeights
        model = GgufWeights(path, manifest['context'], 'none')
        result = trace(model, manifest['prompt_ids'], manifest['steps'])
        del model
    else:
        from dllm.runtime import Runtime
        expected = json.loads((folder/f'{alias}-trace.json').read_text())['result']
        runtime = Runtime({'a': path}, context=manifest['context'], use_graph=False,
                          gpu_models=1, host_models=1, host_gib=2, queue_size=1,
                          queue_timeout=10, quantizations={'a': 'none'})
        try:
            result = http_check(runtime, manifest['prompt_ids'], manifest['steps'], expected)
        finally:
            runtime.close()
    gc.collect()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()
    torch.cuda.empty_cache()
    if peak > cap or torch.cuda.memory_allocated() != 0:
        raise ValueError('allocator budget or cleanup failed')
    verify_files(source)
    save(folder/f'{alias}-{phase}.json', dict(result=result, peak_reserved_bytes=peak,
                                            device=torch.cuda.get_device_name(), torch=torch.__version__))


def execute(folder, invoke=subprocess.run):
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['schema'] != 1 or not 32 <= manifest['context'] <= 128 or not 1 <= manifest['steps'] <= 8:
        raise ValueError('invalid manifest bounds')
    if manifest['allocator_mib'] != 1536 or manifest['timeout_seconds'] != 120:
        raise ValueError('invalid resource bounds')
    if not manifest['prompt_ids'] or len(manifest['prompt_ids'])+manifest['steps'] > manifest['context']:
        raise ValueError('invalid prompt bounds')
    if (folder/'result.json').exists():
        raise ValueError('use a fresh prepared directory for each attempt')
    report = dict(passed=False, hardware_validated=False, performance_qualified=False, completed=[])
    started = time.monotonic()
    save(folder/'result.json', report)
    try:
        for phase in ('trace', 'http'):
            for alias in ('single', 'split'):
                remaining = 120-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('total qualification deadline')
                with (folder/f'{alias}-{phase}.log').open('x') as log:
                    invoke([sys.executable, '-m', 'scripts.experiments.split_gguf', '--worker', str(folder), alias, phase],
                           stdout=log, stderr=log, timeout=remaining, check=True)
                report['completed'].append(f'{alias}-{phase}')
                save(folder/'result.json', report)
            a, b = [json.loads((folder/f'{alias}-{phase}.json').read_text())['result']
                    for alias in ('single', 'split')]
            if a != b:
                raise ValueError(f'{phase} single/split mismatch')
        report.update(passed=True, hardware_validated=True)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        save(folder/'result.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', nargs=3, metavar=('SINGLE', 'FIRST_SHARD', 'OUTPUT'))
    mode.add_argument('--run', type=Path)
    mode.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        prepare(Path(args.prepare[0]), Path(args.prepare[1]), Path(args.prepare[2]))
        return
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('DLLM_SHARED_CUDA_GUARD') != '1':
        parser.error('execute through guard_shared_cuda with one explicit GPU UUID')
    if args.run:
        execute(args.run)
    else:
        worker(Path(args.worker[0]), args.worker[1], args.worker[2])


if __name__ == '__main__':
    main()
