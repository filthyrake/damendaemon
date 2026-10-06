"""Validate a published GGUF against independent dequantization and HF execution.

Transformers and gguf.quants are validation-only here; the production importer
uses neither Transformers execution nor the upstream dequantizer for GPU math.
The host importer may use codec decoding for rare embedding-finiteness checks.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
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


def reference_prefill(reference, tokens, chunk_size=0):
    """Keep HF attention temporaries bounded without substituting its math."""
    if not chunk_size:
        return reference(tokens, use_cache=True)
    past = None
    for start in range(0, tokens.shape[1], chunk_size):
        output = reference(tokens[:, start:start + chunk_size], past_key_values=past, use_cache=True)
        past = output.past_key_values
    return output


def reference_weights(path, rounding='float16'):
    if rounding not in ('float16','kquant-float32'):
        raise ValueError('unsupported reference weight rounding')
    reader = GGUFReader(path)
    state = {}
    architecture = reader.fields['general.architecture'].contents()
    root = {'token_embd.weight':'model.embed_tokens.weight', 'output_norm.weight':'model.norm.weight', 'output.weight':'lm_head.weight'}
    layer = {'attn_norm.weight':'input_layernorm.weight', 'ffn_norm.weight':'post_attention_layernorm.weight',
             'attn_output.weight':'self_attn.o_proj.weight', 'ffn_gate.weight':'mlp.gate_proj.weight',
             'ffn_up.weight':'mlp.up_proj.weight', 'ffn_down.weight':'mlp.down_proj.weight',
             'attn_q_norm.weight':'self_attn.q_norm.weight', 'attn_k_norm.weight':'self_attn.k_norm.weight'}
    for axis in ('q', 'k', 'v'):
        for kind in ('weight', 'bias'): layer[f'attn_{axis}.{kind}'] = f'self_attn.{axis}_proj.{kind}'
    types = {}
    tied_head = None
    for tensor in reader.tensors:
        if tensor.name == 'rope_freqs.weight':
            continue  # HF independently constructs the Llama3 rotary schedule.
        target = root.get(tensor.name)
        if target is None:
            match = re.fullmatch(r'blk\.(\d+)\.(.+)', tensor.name)
            if not match or match[2] not in layer: raise ValueError(tensor.name)
            target = f'model.layers.{match[1]}.' + layer[match[2]]
        value = dequantize(tensor.data, tensor.tensor_type)
        if architecture == 'llama' and tensor.name.endswith(('attn_q.weight','attn_k.weight')):
            key = 'llama.attention.head_count' if tensor.name.endswith('attn_q.weight') else 'llama.attention.head_count_kv'
            heads = reader.fields[key].contents()
            shape = value.shape
            value = value.reshape(heads,shape[0]//heads//2,2,*shape[1:]).swapaxes(1,2).reshape(shape)
        decoded = torch.from_numpy(value.copy())
        if rounding=='kquant-float32' and int(tensor.tensor_type) in (2,8):
            raise ValueError('FP32 K-quant oracle does not cover legacy Q4_0/Q8_0 matrices')
        exact_kquant = rounding=='kquant-float32' and int(tensor.tensor_type) in (12,13,14)
        if tensor.name=='token_embd.weight' and exact_kquant:
            tied_head=decoded
        state[target] = decoded if exact_kquant and tensor.name!='token_embd.weight' else decoded.half()
        types[tensor.tensor_type.name] = types.get(tensor.tensor_type.name, 0) + 1
    if 'lm_head.weight' not in state:
        state['lm_head.weight'] = tied_head if tied_head is not None else state['model.embed_tokens.weight']
    embedding_type = next(t.tensor_type.name for t in reader.tensors if t.name == "token_embd.weight")
    return state, types, embedding_type


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf_config'); p.add_argument('--output', required=True)
    p.add_argument('--benchmark-report',help='validate the exact prompt IDs and output length from an engine comparison')
    p.add_argument('--context',type=int,default=512,help='context capacity; must be at least 512')
    p.add_argument('--extra-prompt-tokens',type=int,nargs='*',default=[],help='additional raw prompt lengths, leaving eight decode positions')
    p.add_argument('--weight-rounding',choices=('float16','kquant-float32'),default='float16',
                   help='explicit coefficient policy; FP32 K projections retain FP16 embedding/vector values')
    p.add_argument('--serial-reference',action='store_true',help='cache independent HF logits, release HF GPU weights, then run dllm')
    p.add_argument('--reference-prefill-chunk-size', type=int, default=0,
                   help='bound independent HF prefill attention memory; zero uses whole prompts')
    p.add_argument('--reference-dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--teacher-trace', help='hold token inputs fixed to a previous report')
    p.add_argument('--extended', action='store_true', help='additional multi-turn, Unicode, code and raw-prefix cases')
    p.add_argument('--window-boundaries', action='store_true', help='include prompts before, at and after the declared sliding window')
    p.add_argument('--expanded-control',action='store_true',help='diagnostic: expand the same coefficients before dllm execution')
    p.add_argument('--require-q4-marlin',action='store_true',help='require normal Q4 Marlin layout replacement')
    p.add_argument('--require-resident-cache', action='store_true', help='fail unless normal resident cache selection is active')
    p.add_argument('--require-q4-small-tiles', action='store_true', help='fail unless normal A2 prompt tile selection is active for this model')
    p.add_argument('--experimental-q4-marlin-symmetric',action='store_true')
    p.add_argument('--experimental-q4-marlin-replace', action='store_true')
    experiments=p.add_mutually_exclusive_group()
    experiments.add_argument('--experimental-q4-marlin', metavar='LIBRARY')
    experiments.add_argument('--experimental-q4-small-tiles', action='store_true')
    experiments.add_argument('--experimental-q4-residual', action='store_true')
    experiments.add_argument('--experimental-residual', action='store_true')
    experiments.add_argument('--experimental-residual-f32', action='store_true')
    experiments.add_argument('--experimental-prefill-f32', action='store_true')
    experiments.add_argument('--experimental-expand-tiles', action='store_true')
    experiments.add_argument('--experimental-dense-cache', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4)
    if args.context < 512:p.error('--context must be at least 512')
    if args.reference_prefill_chunk_size < 0:p.error('--reference-prefill-chunk-size must be nonnegative')
    if args.weight_rounding=='kquant-float32' and args.reference_dtype!='float32':
        p.error('kquant-float32 weights require --reference-dtype float32')
    experiment = None
    if (args.experimental_q4_marlin_replace or args.experimental_q4_marlin_symmetric) and not args.experimental_q4_marlin:
        p.error('--experimental-q4-marlin-replace requires --experimental-q4-marlin')
    if args.experimental_q4_marlin_replace and args.experimental_q4_marlin_symmetric:
        p.error('choose only one Marlin replacement variant')
    if args.experimental_q4_marlin:
        if args.experimental_q4_marlin_replace or args.experimental_q4_marlin_symmetric:
            from experiments.q4_marlin_replace import install_validation_dispatch
        else:
            from experiments.q4_marlin_dispatch import install_validation_dispatch
        experiment = (install_validation_dispatch(args.experimental_q4_marlin,implicit_zero=True)
                      if args.experimental_q4_marlin_symmetric else install_validation_dispatch(args.experimental_q4_marlin))
    if args.experimental_q4_small_tiles:
        from experiments.q4_small_tiles import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_dense_cache:
        from experiments.kquant_dense_cache import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_expand_tiles:
        from experiments.kquant_expand import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_q4_residual:
        from experiments.q4_residual import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_residual:
        from experiments.kquant_residual import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_residual_f32:
        if args.weight_rounding!='kquant-float32':
            p.error('FP32 experiment requires --weight-rounding kquant-float32')
        from experiments.kquant_residual_f32 import install_validation_dispatch
        experiment=install_validation_dispatch()
    if args.experimental_prefill_f32:
        if args.weight_rounding!='kquant-float32':
            p.error('FP32 prefill experiment requires --weight-rounding kquant-float32')
        from experiments.kquant_prefill_f32 import install_validation_dispatch
        experiment=install_validation_dispatch()
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = json.loads(Path(args.teacher_trace).read_text()) if args.teacher_trace else None
    source = GgufWeights(args.gguf, args.context, 'none')
    if args.expanded_control:
        from verify_kquant_runtime import expanded
        source=expanded(source)
    config = AutoConfig.from_pretrained(args.hf_config, local_files_only=True, trust_remote_code=False)
    config.tie_word_embeddings = False
    with no_init_weights():
        reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
    state, formats, embedding_type = reference_weights(args.gguf,args.weight_rounding)
    reference.load_state_dict(state, strict=True, assign=True); del state
    reference.eval().to(device="cuda", dtype=getattr(torch, args.reference_dtype))
    cases = []
    prompts = [source.chat_tokens([{'role':'user','content':text}]) for text in
               ('What is 2 + 2? Answer with one number.', 'Reply in French: Hello, how are you?')]
    base = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
    for length in (129, args.context-7): prompts.append((base * ((length+len(base)-1)//len(base)))[:length])
    if args.extended:
        prompts = [source.chat_tokens(messages) for messages in (
            [{'role':'system','content':'Answer briefly.'}, {'role':'user','content':'What is 17 + 25? Just the number.'}],
            [{'role':'user','content':'Remember the code ZX-914.'}, {'role':'assistant','content':'The code is ZX-914.'},
             {'role':'user','content':'Repeat only the code.'}],
            [{'role':'user','content':'Translate into Spanish: The café opens at eight. 👋'}],
            [{'role':'user','content':'Write a Python function that returns the largest number in a list.'}])]
        corpus = source.raw_tokens('Library record 73 is in cabinet 7. Record 85 is in cabinet 8. Locate the requested record. ')
        for length in (257, args.context-127): prompts.append((corpus * ((length+len(corpus)-1)//len(corpus)))[:length])
    if args.window_boundaries:
        window = source.config.get('sliding_window')
        if not window or args.context < window + 8 or args.benchmark_report:
            p.error('--window-boundaries requires a declared window, context >= window+8, and no benchmark report')
        for length in (window - 1, window, window + 1):
            if length > 0 and length not in [len(ids) for ids in prompts]:
                prompts.append((base * ((length+len(base)-1)//len(base)))[:length])
    if args.extra_prompt_tokens and args.benchmark_report:
        p.error('--extra-prompt-tokens cannot be combined with --benchmark-report')
    if any(length < 1 or length + 7 > args.context for length in args.extra_prompt_tokens):
        p.error('extra prompt lengths must be positive and leave eight decode positions')
    for length in dict.fromkeys(args.extra_prompt_tokens):
        if length not in [len(ids) for ids in prompts]:
            prompts.append((base * ((length+len(base)-1)//len(base)))[:length])
    benchmark=json.loads(Path(args.benchmark_report).read_text()) if args.benchmark_report else None
    steps=8
    if benchmark:
        if args.extended:p.error('--benchmark-report cannot be combined with --extended')
        prompts=[benchmark['prompts'][key] for key in sorted(benchmark['prompts'],key=int)]
        steps=benchmark['output_tokens']
        if type(steps) is not int or not 1<=steps<=256:
            raise ValueError('invalid benchmark output length')
        if any(not ids or len(ids)+steps-1>args.context or any(type(t) is not int or not 0<=t<source.config['vocab_size'] for t in ids) for ids in prompts):
            raise ValueError('invalid benchmark prompt tokens or context budget')
    reference_cache = {}
    if args.serial_reference:
        # Large checkpoints need not hold HF's expanded weights and dllm's
        # packed weights on the GPU together. The oracle still executes independently.
        with torch.inference_mode():
            for graph in (False, True):
                for index, ids in enumerate(prompts):
                    previous = next(c for c in teacher["cases"] if c["graph"] == graph and c["case"] == index) if teacher else None
                    if previous and previous['prompt_tokens'] != len(ids):
                        raise ValueError('teacher trace uses a different prompt/context length')
                    expected = reference_prefill(reference, torch.tensor([ids], device='cuda'), args.reference_prefill_chunk_size)
                    trace = []
                    for step in range(steps):
                        target = expected.logits[:, -1, :].cpu()
                        trace.append(target)
                        token = previous["reference_tokens"][step] if previous else target.argmax(-1).item()
                        if step < steps-1:
                            expected = reference(torch.tensor([[token]], device='cuda'), past_key_values=expected.past_key_values, use_cache=True)
                    reference_cache[graph, index] = trace
                    del expected
        del reference
        gc.collect(); empty_cache()
    q4_prompt_policy = None
    with torch.inference_mode():
        for graph in (False, True):
            executor = make_executor(source, use_graph=graph)
            if args.require_q4_small_tiles:
                from dllm import q4_prefill_policy
                weight = executor.w['0.qkv.weight']
                selected = (weight.dtype == torch.uint8 and executor.compute_dtype == torch.float16
                            and q4_prefill_policy.use_small_tile(64, weight.shape[0], weight.shape[1]*2, weight.device))
                if not selected:
                    raise ValueError('normal Q4 prompt tile policy was not selected')
                q4_prompt_policy = dict(selected=True, source_sha256=hashlib.sha256(
                    Path(q4_prefill_policy.__file__).read_bytes()).hexdigest())
            if args.require_q4_marlin:
                assert '+cuda-q4-marlin' in executor.kernel_backend,executor.kernel_backend
                assert len(executor.quant_linear.q4_marlin)==129
                assert executor.packed_layout_overhead_bytes==111099904
            if args.require_resident_cache and not getattr(executor, 'packed_cache_bytes', 0):
                raise ValueError('normal resident cache was not selected')
            if args.experimental_dense_cache:
                experiment['resident_cache_bytes'] = executor.quant_linear.experimental_dense_cache_bytes
                if not experiment['resident_cache_bytes']:
                    raise ValueError('dense-cache experiment selected no projections')
            if args.experimental_q4_marlin:
                experiment['additional_prepared_bytes'] = executor.quant_linear.experimental_q4_marlin_bytes
                if args.experimental_q4_marlin_replace or args.experimental_q4_marlin_symmetric:
                    experiment['prepared_weight_bytes'] = executor.quant_linear.experimental_q4_marlin_total_bytes
                    experiment['replaced_canonical_bytes'] = executor.quant_linear.experimental_q4_marlin_replaced_bytes
                if not experiment.get('prepared_weight_bytes',experiment['additional_prepared_bytes']):
                    raise ValueError('Q4 Marlin experiment selected no projections')
            if (args.experimental_residual_f32 or args.experimental_prefill_f32) and executor.compute_dtype!=torch.float32:
                raise ValueError('FP32 experiment requires an FP32 executor policy')
            precision = dict(matrix_dtype=str(executor.compute_dtype), residual_dtype=str(executor.residual_dtype),
                             resident_projection_cache_bytes=getattr(executor, 'packed_cache_bytes', 0),
                             nonlinear_precise=executor.precise, rotary_precise=executor.rotary_precise, backend=executor.kernel_backend)
            for index, ids in enumerate(prompts):
                expected = None if args.serial_reference else reference_prefill(reference, torch.tensor([ids], device='cuda'), args.reference_prefill_chunk_size)
                actual = executor.prefill(ids)
                errors = []; output = []; actual_tokens = []; margins = []; mismatches = []; forced = []; eos_step = None
                previous = next(c for c in teacher["cases"] if c["graph"] == graph and c["case"] == index) if teacher else None
                if previous and previous['prompt_tokens'] != len(ids):
                    raise ValueError('teacher trace uses a different prompt/context length')
                for step in range(steps):
                    target = reference_cache[graph, index][step].to('cuda') if args.serial_reference else expected.logits[:, -1, :]
                    error = actual.float()-target.float()
                    absolute = error.abs().max().item()
                    relative = (error.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item()
                    errors.append(dict(step=step, max_absolute=absolute, relative_rms=relative))
                    token = target.argmax(-1).item(); output.append(token)
                    actual_tokens.append(actual.argmax(-1).item())
                    best=target.float().topk(2,dim=-1).values
                    margins.append((best[0,0]-best[0,1]).item())
                    if eos_step is None and token in source.eos: eos_step = step
                    next_input = previous["reference_tokens"][step] if previous else token
                    forced.append(next_input)
                    if token != actual.argmax(-1).item(): mismatches.append(step)
                    if step < steps-1:
                        if not args.serial_reference:
                            expected = reference(torch.tensor([[next_input]], device='cuda'), past_key_values=expected.past_key_values, use_cache=True)
                        actual, _ = executor.decode(next_input)
                cases.append(dict(graph=graph, case=index, prompt_tokens=len(ids), errors=errors,
                                  greedy_mismatches=mismatches, reference_tokens=output, actual_tokens=actual_tokens, reference_top_two_margins=margins, teacher_tokens=forced, eos_step=eos_step))
                del expected, actual, target, error
            if args.experimental_q4_marlin:
                calls = executor.quant_linear.experimental_q4_marlin_calls
                if not calls:
                    raise ValueError('Q4 Marlin experiment executed no projections')
                experiment.setdefault('executions', []).append(dict(graph=graph, dispatch_calls=calls))
            del executor
            gc.collect(); empty_cache()
    with Path(args.gguf).open('rb') as file: sha = hashlib.file_digest(file, 'sha256').hexdigest()
    passed = all(not c['greedy_mismatches'] and all(e['max_absolute'] < .5 and e['relative_rms'] < .01 for e in c['errors']) for c in cases)
    residual_variable=('DLLM_KQUANT_RESIDUAL_F32_LIBRARY' if 'cuda-kquant-residual-f32' in precision['backend']
                       else 'DLLM_KQUANT_RESIDUAL_LIBRARY')
    residual_path=os.environ.get(residual_variable)
    residual_selected='cuda-kquant-residual' in precision['backend']
    residual_library=None
    if residual_path:
        residual_library=dict(path=residual_path,selected=residual_selected,
            sha256=hashlib.sha256(Path(residual_path).read_bytes()).hexdigest() if residual_selected else None)
    prefill_kernel=None
    if 'compensated-kquant-prefill-f32' in precision['backend']:
        from dllm import kquant_prefill_kernels
        prefill_kernel=hashlib.sha256(Path(kquant_prefill_kernels.__file__).read_bytes()).hexdigest()
    q4_path=os.environ.get('DLLM_Q4_CUDA_LIBRARY')
    q4_library=dict(path=q4_path, selected='+cuda-gemv' in precision['backend'],
                    sha256=hashlib.sha256(Path(q4_path).read_bytes()).hexdigest()) if q4_path else None
    q4_residual_path=os.environ.get('DLLM_Q4_RESIDUAL_LIBRARY')
    q4_residual_library=dict(path=q4_residual_path,selected='+cuda-q4-residual' in precision['backend'],
        sha256=hashlib.sha256(Path(q4_residual_path).read_bytes()).hexdigest()) if q4_residual_path else None
    kquant_path=os.environ.get('DLLM_KQUANT_CUDA_LIBRARY')
    kquant_library=dict(path=kquant_path,sha256=hashlib.sha256(Path(kquant_path).read_bytes()).hexdigest()) if kquant_path else None
    tokenizer_runtime=dict(kind=type(source.tokenizer).__name__)
    if hasattr(source.tokenizer,'native_model'):
        tokenizer_runtime.update(sentencepiece=version('sentencepiece'),protobuf=version('protobuf'))
    report = dict(tokenizer_runtime=tokenizer_runtime, native_kquant_library=kquant_library, native_q4_residual_library=q4_residual_library, native_q4_library=q4_library, serial_reference=args.serial_reference, reference_cache_bytes=sum(t.numel()*t.element_size() for trace in reference_cache.values() for t in trace), production_prefill_kernel_sha256=prefill_kernel, passed=passed, device=device_name(), torch=torch.__version__, gguf=args.gguf, sha256=sha,
                  precision=precision, q4_prompt_policy=q4_prompt_policy, context=args.context, suite="benchmark" if benchmark else "extended" if args.extended else "initial",
                  benchmark_report=args.benchmark_report, steps=steps,
                  reference=f'independent gguf.quants dequantization into HF eager {config.model_type} execution', reference_dtype=args.reference_dtype,
                  reference_prefill_chunk_size=args.reference_prefill_chunk_size,
                  window_boundary_validation=args.window_boundaries, sliding_window=source.config.get('sliding_window'),
                  weight_rounding=args.weight_rounding, tf32=False, formats=formats, embedding_type=embedding_type,
                  teacher_trace=args.teacher_trace,expanded_control=args.expanded_control,
                  host_bytes=source.bytes, experiment=experiment, production_residual_library=residual_library, cases=cases)
    marlin_path=os.environ.get('DLLM_Q4_MARLIN_LIBRARY')
    if marlin_path:
        report['native_q4_marlin_library']=dict(path=marlin_path,selected='+cuda-q4-marlin' in precision['backend'],
            sha256=hashlib.sha256(Path(marlin_path).read_bytes()).hexdigest())
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=passed, sha256=sha, formats=formats,
        max_absolute=max(e['max_absolute'] for c in cases for e in c['errors']),
        max_relative_rms=max(e['relative_rms'] for c in cases for e in c['errors']),
        greedy_mismatches=sum(len(c['greedy_mismatches']) for c in cases))), flush=True)
    assert passed, 'published GGUF numerical gate failed'


if __name__ == '__main__': main()
