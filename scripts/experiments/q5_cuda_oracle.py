"""Fixed CUDA FP16 reference with CPU-offloaded weights; owned shared run only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.experiments.q5_precision_replay import CHECKPOINT_SHA256, checkpoint, teacher_case
from scripts.experiments.split_gguf import save


def require_guard():
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (os.environ.get('DAMENDAEMON_SHARED_CUDA_GUARD') != '1' or
            not uuid.startswith('GPU-') or ',' in uuid):
        raise ValueError('requires shared CUDA guard and one explicit GPU UUID')


def verify_cuda_output(output):
    """Hooks run after Accelerate's forward wrapper, not before weight transfer."""
    import torch
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.device.type != 'cuda':
        raise ValueError('reference arithmetic did not return CUDA tensors')


def export(model_path, config_path, teacher_path, folder):
    require_guard()
    import numpy as np
    import torch
    import accelerate
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from damendaemon.gguf_model import GgufWeights
    from damendaemon.rocm_policy import host_available
    from scripts.verify_gguf_reference import reference_weights
    if accelerate.__version__ != '1.10.0':
        raise ValueError('oracle requires isolated Accelerate 1.10.0')
    if host_available() < 32*1024**3:
        raise MemoryError('oracle requires 32GiB available host memory')
    if not torch.cuda.is_available() or torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('one NVIDIA GPU required')
    folder.mkdir()
    report = dict(completed=False, checkpoint_sha256=CHECKPOINT_SHA256,
                  reference_device='cuda', weight_rounding='float16', reference_dtype='float16',
                  offload='accelerate.cpu_offload', accelerate=accelerate.__version__,
                  torch=torch.__version__, device=torch.cuda.get_device_name(), context=64,
                  steps=8, historical_gpu_gate_replaced=False, oracles={})
    save(folder/'oracle.json', report)
    hooks = []
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(1536*1024**2/torch.cuda.get_device_properties(0).total_memory)
        checkpoint(model_path)
        teacher = teacher_case(json.loads(teacher_path.read_text()))
        source = GgufWeights(model_path,64,'none')
        ids = source.chat_tokens([dict(role='user',content='What is 2 + 2? Answer with one number.')])
        if len(ids) != 48:
            raise ValueError('historical prompt length changed')
        del source
        gc.collect()
        config = AutoConfig.from_pretrained(config_path,local_files_only=True,trust_remote_code=False)
        config.tie_word_embeddings = False
        with no_init_weights():
            reference = AutoModelForCausalLM.from_config(config,torch_dtype=torch.float16,attn_implementation='eager')
        state, formats, embedding = reference_weights(str(model_path),'float16')
        reference.load_state_dict(state,strict=True,assign=True)
        del state
        reference.eval().to(dtype=torch.float16)
        accelerate.cpu_offload(reference,execution_device=torch.device('cuda:0'),offload_buffers=True)
        calls = {}
        # Check both learned projections and complete decoder outputs, covering
        # attention/activation arithmetic as well as final vocabulary projection.
        watched = {name:module for name,module in reference.named_modules()
                   if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)) or
                   module.__class__.__name__ == 'LlamaDecoderLayer'}
        if not watched:
            raise ValueError('no reference CUDA arithmetic modules found')
        def hook(name):
            def checked(module, inputs, output):
                verify_cuda_output(output)
                calls[name] = calls.get(name,0)+1
            return checked
        for name,module in watched.items():
            hooks.append(module.register_forward_hook(hook(name)))
        report.update(prompt_ids=ids,teacher_tokens=teacher,formats=formats,embedding=embedding,
                      hf_config_sha256=hashlib.sha256((config_path/'config.json').read_bytes()).hexdigest())
        save(folder/'oracle.json',report)
        values = []
        with torch.inference_mode():
            value = reference(torch.tensor([ids],device='cuda'),use_cache=True)
            for step,token in enumerate(teacher):
                verify_cuda_output(value.logits)
                values.append(value.logits[0,-1].float().cpu().numpy().copy())
                if step < 7:
                    value = reference(torch.tensor([[token]],device='cuda'),past_key_values=value.past_key_values,use_cache=True)
        if set(calls) != set(watched) or any(count != 8 for count in calls.values()):
            raise ValueError('incomplete reference CUDA execution coverage')
        result = np.stack(values)
        if result.shape != (8,128256) or not np.isfinite(result).all():
            raise ValueError('invalid CUDA oracle logits')
        np.save(folder/'cuda_float16.npy',result,allow_pickle=False)
        report['oracles']['cuda_float16'] = dict(sha256=hashlib.sha256((folder/'cuda_float16.npy').read_bytes()).hexdigest())
        report.update(completed=True,cuda_module_calls=calls,seconds=time.monotonic()-started,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        for handle in hooks:
            handle.remove()
        save(folder/'oracle.json',report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','config','teacher','folder'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    export(args.model,args.config,args.teacher,args.folder)


if __name__ == '__main__':
    main()
