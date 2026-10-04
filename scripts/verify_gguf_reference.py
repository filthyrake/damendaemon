"""Validate a published GGUF against independent dequantization and HF execution.

Transformers and gguf.quants are validation-only here; the production importer
uses neither Transformers execution nor the upstream dequantizer.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re

import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from dllm.gguf_model import GgufWeights
from dllm.runtime import make_executor
from dllm.device import empty_cache, device_name


def reference_weights(path):
    reader = GGUFReader(path)
    state = {}
    root = {'token_embd.weight':'model.embed_tokens.weight', 'output_norm.weight':'model.norm.weight', 'output.weight':'lm_head.weight'}
    layer = {'attn_norm.weight':'input_layernorm.weight', 'ffn_norm.weight':'post_attention_layernorm.weight',
             'attn_output.weight':'self_attn.o_proj.weight', 'ffn_gate.weight':'mlp.gate_proj.weight',
             'ffn_up.weight':'mlp.up_proj.weight', 'ffn_down.weight':'mlp.down_proj.weight',
             'attn_q_norm.weight':'self_attn.q_norm.weight', 'attn_k_norm.weight':'self_attn.k_norm.weight'}
    for axis in ('q', 'k', 'v'):
        for kind in ('weight', 'bias'): layer[f'attn_{axis}.{kind}'] = f'self_attn.{axis}_proj.{kind}'
    types = {}
    for tensor in reader.tensors:
        target = root.get(tensor.name)
        if target is None:
            match = re.fullmatch(r'blk\.(\d+)\.(.+)', tensor.name)
            if not match or match[2] not in layer: raise ValueError(tensor.name)
            target = f'model.layers.{match[1]}.' + layer[match[2]]
        value = dequantize(tensor.data, tensor.tensor_type)
        state[target] = torch.from_numpy(value.copy()).to(torch.float16)
        types[tensor.tensor_type.name] = types.get(tensor.tensor_type.name, 0) + 1
    if 'lm_head.weight' not in state: state['lm_head.weight'] = state['model.embed_tokens.weight']
    embedding_type = next(t.tensor_type.name for t in reader.tensors if t.name == "token_embd.weight")
    return state, types, embedding_type


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf_config'); p.add_argument('--output', required=True)
    p.add_argument('--reference-dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--teacher-trace', help='hold token inputs fixed to a previous report')
    args = p.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = json.loads(Path(args.teacher_trace).read_text()) if args.teacher_trace else None
    source = GgufWeights(args.gguf, 512, 'none')
    config = AutoConfig.from_pretrained(args.hf_config, local_files_only=True, trust_remote_code=False)
    config.tie_word_embeddings = False
    with no_init_weights():
        reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
    state, formats, embedding_type = reference_weights(args.gguf)
    reference.load_state_dict(state, strict=True, assign=True); del state
    reference.eval().to(device="cuda", dtype=getattr(torch, args.reference_dtype))
    cases = []
    prompts = [source.chat_tokens([{'role':'user','content':text}]) for text in
               ('What is 2 + 2? Answer with one number.', 'Reply in French: Hello, how are you?')]
    base = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
    for length in (129, 505): prompts.append((base * ((length+len(base)-1)//len(base)))[:length])
    with torch.inference_mode():
        for graph in (False, True):
            executor = make_executor(source, use_graph=graph)
            for index, ids in enumerate(prompts):
                expected = reference(torch.tensor([ids], device='cuda'), use_cache=True)
                actual = executor.prefill(ids)
                errors = []; output = []; mismatches = []; forced = []; eos_step = None
                previous = next(c for c in teacher["cases"] if c["graph"] == graph and c["case"] == index) if teacher else None
                for step in range(8):
                    target = expected.logits[:, -1, :]
                    error = actual.float()-target.float()
                    absolute = error.abs().max().item()
                    relative = (error.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item()
                    errors.append(dict(step=step, max_absolute=absolute, relative_rms=relative))
                    token = target.argmax(-1).item(); output.append(token)
                    if eos_step is None and token in source.eos: eos_step = step
                    next_input = previous["reference_tokens"][step] if previous else token
                    forced.append(next_input)
                    if token != actual.argmax(-1).item(): mismatches.append(step)
                    if step < 7:
                        expected = reference(torch.tensor([[next_input]], device='cuda'), past_key_values=expected.past_key_values, use_cache=True)
                        actual, _ = executor.decode(next_input)
                cases.append(dict(graph=graph, case=index, prompt_tokens=len(ids), errors=errors,
                                  greedy_mismatches=mismatches, reference_tokens=output, teacher_tokens=forced, eos_step=eos_step))
                del expected, actual, target, error
            del executor
            gc.collect(); empty_cache()
    with Path(args.gguf).open('rb') as file: sha = hashlib.file_digest(file, 'sha256').hexdigest()
    passed = all(not c['greedy_mismatches'] and all(e['max_absolute'] < .5 and e['relative_rms'] < .01 for e in c['errors']) for c in cases)
    report = dict(passed=passed, device=device_name(), torch=torch.__version__, gguf=args.gguf, sha256=sha,
                  reference='independent gguf.quants dequantization into HF eager Qwen execution', reference_dtype=args.reference_dtype,
                  weight_rounding='float16', tf32=False, formats=formats, embedding_type=embedding_type,
                  teacher_trace=args.teacher_trace,
                  host_bytes=source.bytes, cases=cases)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=passed, sha256=sha, formats=formats,
        max_absolute=max(e['max_absolute'] for c in cases for e in c['errors']),
        max_relative_rms=max(e['relative_rms'] for c in cases for e in c['errors']),
        greedy_mismatches=sum(len(c['greedy_mismatches']) for c in cases))), flush=True)
    assert passed, 'published GGUF numerical gate failed'


if __name__ == '__main__': main()
