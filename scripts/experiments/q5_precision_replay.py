"""Small fixed-token diagnostic; CPU oracles do not replace historical GPU gates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from scripts.experiments.split_gguf import digest, save

CHECKPOINT_SHA256 = 'e593d886d3cbdfed7a7c3c650866a7506d29f2e7d16218409316773f47c91db7'


def checkpoint(path):
    with path.open('rb') as file:
        actual = digest(file, path.stat().st_size)
    if actual != CHECKPOINT_SHA256:
        raise ValueError('requires the pinned historical Q5 checkpoint')


def teacher_case(report):
    if report.get('sha256') != CHECKPOINT_SHA256:
        raise ValueError('teacher checkpoint mismatch')
    case = next(c for c in report['cases'] if c['case'] == 0 and c['graph'] is False)
    tokens = case['reference_tokens']
    if len(tokens) != 8 or any(type(t) is not int or not 0 <= t < 128256 for t in tokens):
        raise ValueError('expected eight valid teacher tokens')
    if case['prompt_tokens'] != 48:
        raise ValueError('historical prompt length changed')
    return tokens


def metrics(actual, expected, teacher):
    import numpy as np
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[0] != len(teacher):
        raise ValueError('oracle shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite oracle/candidate')
    rows = []
    for step, (a, b) in enumerate(zip(actual, expected)):
        difference = a.astype('float64')-b.astype('float64')
        rms = float(np.sqrt(np.mean(difference**2)/max(float(np.mean(b.astype('float64')**2)), 1e-12)))
        rows.append(dict(step=step, relative_rms=rms, max_absolute=float(np.max(np.abs(difference))),
                         candidate_token=int(a.argmax()), oracle_token=int(b.argmax()),
                         teacher_token=teacher[step]))
    return dict(passed=all(r['relative_rms'] < .01 and r['max_absolute'] < .5 and
                           r['candidate_token'] == r['oracle_token'] for r in rows), errors=rows)


def export_cpu(model_path, config_path, teacher_path, folder):
    import numpy as np
    import torch
    from dllm.rocm_policy import host_available
    from dllm.gguf_model import GgufWeights
    from scripts.verify_gguf_reference import reference_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU export requires CUDA_VISIBLE_DEVICES empty')
    if host_available() < 32*1024**3:
        raise MemoryError('CPU oracle requires32GiB available host memory')
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    torch.set_num_threads(1)
    checkpoint(model_path)
    teacher = teacher_case(json.loads(teacher_path.read_text()))
    source = GgufWeights(model_path, 64, 'none')
    ids = source.chat_tokens([dict(role='user', content='What is 2 + 2? Answer with one number.')])
    if len(ids) != 48 or len(ids)+len(teacher) > 64:
        raise ValueError('prompt/context mismatch')
    del source
    gc.collect()
    folder.mkdir()
    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=False)
    with no_init_weights():
        reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
    state, formats, embedding = reference_weights(str(model_path), 'float16')
    reference.load_state_dict(state, strict=True, assign=True)
    del state
    reference.eval()
    report = dict(checkpoint_sha256=CHECKPOINT_SHA256, prompt_ids=ids, teacher_tokens=teacher,
                  context=64, steps=8, reference_device='cpu', weight_rounding='float16',
                  formats=formats, embedding=embedding, torch=torch.__version__,
                  hf_config_sha256=hashlib.sha256((config_path/'config.json').read_bytes()).hexdigest(),
                  historical_gpu_gate_replaced=False, oracles={})
    save(folder/'oracle.json', report)
    for dtype in ('float16', 'float32'):
        started = time.monotonic()
        reference.to(device='cpu', dtype=getattr(torch, dtype))
        logits = []
        with torch.inference_mode():
            value = reference(torch.tensor([ids]), use_cache=True)
            for step, token in enumerate(teacher):
                logits.append(value.logits[0, -1].float().numpy().copy())
                if step+1 < len(teacher):
                    value = reference(torch.tensor([[token]]), past_key_values=value.past_key_values, use_cache=True)
        result = np.stack(logits)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite CPU oracle')
        np.save(folder/f'{dtype}.npy', result, allow_pickle=False)
        report['oracles'][dtype] = dict(seconds=time.monotonic()-started,
            sha256=hashlib.sha256((folder/f'{dtype}.npy').read_bytes()).hexdigest())
        del value, logits
        gc.collect()
        save(folder/'oracle.json', report)


def oracle_kinds(oracle):
    if oracle.get('reference_device') == 'cpu' and set(oracle.get('oracles', {})) == {'float16', 'float32'}:
        return ('float16', 'float32')
    if (oracle.get('reference_device') == 'cuda' and oracle.get('completed') is True
            and oracle.get('offload') == 'accelerate.cpu_offload'
            and set(oracle.get('oracles', {})) == {'cuda_float16'}):
        return ('cuda_float16',)
    raise ValueError('unsupported or incomplete oracle')


def replay(model_path, folder, precision, output):
    import numpy as np
    import torch
    from dllm.gguf_model import GgufWeights
    from dllm.runtime import make_executor
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('DLLM_SHARED_CUDA_GUARD') != '1':
        raise ValueError('requires shared CUDA guard and explicit GPU UUID')
    if output.exists():
        raise ValueError('new output path required')
    checkpoint(model_path)
    oracle = json.loads((folder/'oracle.json').read_text())
    if oracle['checkpoint_sha256'] != CHECKPOINT_SHA256 or oracle['context'] != 64 or oracle['steps'] != 8:
        raise ValueError('oracle identity/bounds mismatch')
    kinds = oracle_kinds(oracle)
    for dtype in kinds:
        if hashlib.sha256((folder/f'{dtype}.npy').read_bytes()).hexdigest() != oracle['oracles'][dtype]['sha256']:
            raise ValueError('oracle payload changed')
    if not torch.cuda.is_available() or torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(1536*1024**2/torch.cuda.get_device_properties(0).total_memory)
    os.environ['DLLM_LLAMA_PRECISION'] = precision
    report = dict(completed=False, precision=precision, historical_gpu_gate_replaced=False,
                  checkpoint_sha256=CHECKPOINT_SHA256, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, context=64, steps=8)
    save(output, report)
    executor = None
    try:
        model = GgufWeights(model_path,64,'none')
        ids = model.chat_tokens([dict(role='user',content='What is 2 + 2? Answer with one number.')])
        if ids != oracle['prompt_ids']:
            raise ValueError('candidate prompt differs from oracle')
        executor = make_executor(model,use_graph=False)
        report['kernel_backend'] = executor.kernel_backend
        values = []
        with torch.inference_mode():
            logits = executor.prefill(ids)
            for step, token in enumerate(oracle['teacher_tokens']):
                values.append(logits.detach().float().cpu().numpy().reshape(-1).copy())
                if step < 7:
                    logits,_ = executor.decode(token)
        actual = np.stack(values)
        report['comparisons'] = {dtype:metrics(actual,np.load(folder/f'{dtype}.npy',allow_pickle=False),
                                              oracle['teacher_tokens']) for dtype in kinds}
        report['oracle_device'] = oracle['reference_device']
        report['oracle_manifest_sha256'] = hashlib.sha256((folder/'oracle.json').read_bytes()).hexdigest()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['completed'] = True
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        if executor is not None and hasattr(executor,'close'):
            executor.close()
        save(output,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    cpu = sub.add_parser('export-cpu')
    for name in ('model','config','teacher','folder'):cpu.add_argument(name,type=Path)
    gpu = sub.add_parser('replay')
    for name in ('model','folder','output'):gpu.add_argument(name,type=Path)
    gpu.add_argument('--precision',choices=('mixed','float32'),required=True)
    args = parser.parse_args()
    if args.mode == 'export-cpu':export_cpu(args.model,args.config,args.teacher,args.folder)
    else:replay(args.model,args.folder,args.precision,args.output)


if __name__ == '__main__':main()
