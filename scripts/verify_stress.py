"""Long-prefix, split-boundary and rolling-slot checks against an independent decoder."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from damendaemon.batch import BatchExecutor
from damendaemon.qwen2 import ModelWeights
from damendaemon.config import decoder_config
from damendaemon.runtime import require_gpu, make_executor
from reference_model import load_reference
from reference_trace import save_trace
from damendaemon.device import selected_device, synchronize, empty_cache, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument("--reference-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument('--reference-quantization', choices=('none', 'q8_0','q4_0'), default='none')
    parser.add_argument('--prefill-chunk-size',type=int,default=0)
    parser.add_argument('--tiny-prefixes',action='store_true',
                        help='also qualify single and packed 1..16-token prompts with subsequent decode')
    parser.add_argument('--reference-only', action='store_true', help='generate a reference trace without a damendaemon GPU executor')
    parser.add_argument('--golden-output', help='save teacher-forced inputs and FP32 reference logits for cross-environment replay')
    parser.add_argument('--reference-attention', choices=('sdpa', 'eager'), default='sdpa')
    packed_options = parser.add_mutually_exclusive_group()
    packed_options.add_argument('--packed-cpu-reference', action='store_true', help='FP32 CPU reference with per-operation upstream GGUF expansion')
    packed_options.add_argument('--packed-gpu-reference', action='store_true', help='FP32 GPU reference with independent Torch expansion of upstream GGUF-packed weights')
    parser.add_argument('--reference-device', choices=('cpu', 'cuda', 'mps'))
    args = parser.parse_args()
    if args.prefill_chunk_size<0:parser.error('invalid prefill chunk size')
    if args.cpu_threads < 1:parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    reference_device = args.reference_device or selected_device()
    if args.packed_cpu_reference and (not args.reference_only or reference_device != 'cpu' or args.reference_dtype != 'float32'):
        parser.error('--packed-cpu-reference requires --reference-only --reference-device cpu --reference-dtype float32')
    if args.packed_gpu_reference and (not args.reference_only or reference_device != 'cuda' or args.reference_dtype != 'float32'):
        parser.error('--packed-gpu-reference requires --reference-only --reference-device cuda --reference-dtype float32')
    if args.packed_gpu_reference:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.context < 512:
        parser.error("context must be at least 512 for boundary tests")
    require_gpu()
    if args.reference_only and not args.golden_output:
        parser.error("--reference-only requires --golden-output")
    tokenizer = AutoTokenizer.from_pretrained(args.path, local_files_only=True, trust_remote_code=False)
    if args.reference_only:
        config = decoder_config(json.loads((Path(args.path) / 'config.json').read_text()))
        if args.context > config['max_position_embeddings']:
            parser.error('context exceeds model limit')
        model = SimpleNamespace(config=config, raw_tokens=lambda text: tokenizer.encode(text, add_special_tokens=True),
            chat_tokens=lambda messages, **options:
            tokenizer.apply_chat_template(messages, return_dict=False, add_generation_prompt=True, **options))
    else:
        model = ModelWeights(args.path, args.context)
    reference = load_reference(args.path, reference_quantization=args.reference_quantization,
        packed_cpu=args.packed_cpu_reference, packed_gpu=args.packed_gpu_reference, local_files_only=True,
        trust_remote_code=False, torch_dtype=getattr(torch, args.reference_dtype), attn_implementation=args.reference_attention).to(reference_device).eval()
    logit_options = {} if model.config['model_type'] in ('gpt2','bloom') else {'logits_to_keep':1}
    executor = None if args.reference_only else make_executor(model, max_batch=4)
    def native_prefill(prompts,slots):
        if not args.prefill_chunk_size:
            return executor.prefill_batch(prompts,slots)
        values=[]
        for prompt,slot in zip(prompts,slots):
            for start in range(0,len(prompt),args.prefill_chunk_size):
                output=executor.prefill_chunk(prompt[start:start+args.prefill_chunk_size],slot,start,
                    final=start+args.prefill_chunk_size>=len(prompt))
            values.append(output)
        return torch.cat(values)
    cache_lengths = executor.lengths if executor else [0]*4
    checks, template_checks, states, timings = [], [], {}, []
    trace = []
    reference_targets = 0
    reference_started = time.perf_counter()
    trace_metadata = dict(path=args.path, context=args.context,
        reference_dtype=args.reference_dtype, reference_torch=str(torch.__version__),
        cpu_threads=args.cpu_threads, packed_cpu_reference=args.packed_cpu_reference,
        packed_gpu_reference=args.packed_gpu_reference, reference_device=reference_device,
        packed_gpu_expansion='torch' if args.packed_gpu_reference else None,
        reference_quantization=args.reference_quantization, reference_attention=args.reference_attention)

    def checkpoint():
        # Publish completed events only. Never add snapshot I/O to GPU timing
        # runs, and never let an incomplete trace masquerade as the final gate.
        if args.reference_only:
            save_trace(args.golden_output, trace_metadata, trace, complete=False)

    records = 200
    while True:
        documents = ['Record %d: The parcel is stored in cabinet %d, drawer %d.\n' %
                     (i, i % 11, i % 7) for i in range(records)]
        corpus = tokenizer.encode(''.join(documents), add_special_tokens=False)
        if len(corpus) >= args.context + 64:
            break
        records *= 2

    def compare(actual, target, phase, slot):
        nonlocal reference_targets
        if actual is None:
            if not torch.isfinite(target).all().item():
                raise ValueError('reference produced non-finite logits')
            reference_targets += 1
            print(f'reference target={reference_targets} phase={phase} slot={slot} '
                  f'elapsed={time.perf_counter()-reference_started:.1f}s', file=sys.stderr, flush=True)
            return
        actual, target = actual.float().to(target.device), target.float()
        error = (actual - target).abs()
        relative = (error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-6)).item()
        absolute = error.max().item()
        match = actual.argmax().item() == target.argmax().item()
        record = {'phase': phase, 'slot': slot, 'greedy_match': match,
                  'relative_rms': relative, 'max_absolute': absolute,
                  'passed': match and relative < .01 and absolute < .5}
        checks.append(record)

    @torch.inference_mode()
    def prefill(lengths, slots):
        print(f'prefill lengths={lengths} slots={slots}', file=sys.stderr, flush=True)
        prompts = [corpus[slot*13:slot*13+length] for slot, length in zip(slots, lengths)]
        assert all(len(ids) == length for ids, length in zip(prompts, lengths))
        synchronize()
        start = time.perf_counter()
        actual = native_prefill(prompts, slots) if executor else None
        synchronize()
        if executor is not None:
            timings.append({'kind': 'batched_prefill', 'lengths': lengths, 'seconds': time.perf_counter()-start})
        event = {'kind': 'prefill', 'prompts': prompts, 'slots': slots, 'targets': []}
        trace.append(event)
        for row, (slot, ids) in enumerate(zip(slots, prompts)):
            ref = reference(torch.tensor([ids], device=reference_device), use_cache=True, **logit_options)
            target = ref.logits[:, -1]
            event['targets'].append(target.detach().float().cpu())
            compare(actual[row] if actual is not None else None, target, 'prefill:'+str(len(ids)), slot)
            if executor is None:
                cache_lengths[slot] = len(ids)
            states[slot] = (ref.past_key_values, target.argmax().item())
        checkpoint()

    @torch.inference_mode()
    def decode(slots, steps):
        for step in range(steps):
            tokens = [states[slot][1] for slot in slots]
            actual = executor.decode_batch(tokens, slots)[0] if executor else None
            if executor is None:
                for slot in slots:
                    cache_lengths[slot] += 1
            event = {'kind': 'decode', 'tokens': tokens, 'slots': slots, 'targets': []}
            trace.append(event)
            for row, (slot, token) in enumerate(zip(slots, tokens)):
                ref = reference(torch.tensor([[token]], device=reference_device),
                    past_key_values=states[slot][0], use_cache=True, **logit_options)
                target = ref.logits[:, -1]
                event['targets'].append(target.detach().float().cpu())
                compare(actual[row] if actual is not None else None, target, f'decode:{cache_lengths[slot]}', slot)
                states[slot] = (ref.past_key_values, target.argmax().item())
            checkpoint()

    # Real templates, including Unicode, assistant turns, and Qwen3 reasoning controls.
    modes = (None, False, True) if model.config['model_type'] in ('qwen3', 'qwen3_5_text') else (None,)
    messages = [{'role': 'system', 'content': 'Answer briefly.'},
                {'role': 'user', 'content': 'こんにちは — café 🦙. What is 7 × 8?'},
                {'role': 'assistant', 'content': '56.'},
                {'role': 'user', 'content': 'Return it as JSON.'}]
    for mode in modes:
        options = {} if mode is None else {'enable_thinking': mode}
        if tokenizer.chat_template is None:
            text = 'def binary_search(values, target):\n    # café: sorted input\n    '
            ids = model.raw_tokens(text)
            assert ids == tokenizer.encode(text, add_special_tokens=True)
        else:
            ids = model.chat_tokens(messages, **options)
            assert ids == tokenizer.apply_chat_template(messages, return_dict=False, add_generation_prompt=True, **options)
        with torch.inference_mode():
            actual = native_prefill([ids], [0]) if executor else None
            ref = reference(torch.tensor([ids], device=reference_device), use_cache=True, **logit_options)
            compare(actual, ref.logits[:, -1], f'template:{mode}', 0)
            trace.append({'kind': 'prefill', 'prompts': [ids], 'slots': [0],
                          'targets': [ref.logits[:, -1].detach().float().cpu()]})
            checkpoint()
        template_checks.append({'input_kind': 'raw' if tokenizer.chat_template is None else 'chat',
                                'enable_thinking': mode, 'tokens': len(ids), 'runtime_compared': executor is not None})
    del ref
    prefill([31, 255, 257, min(1023,args.context-16)], [3, 0, 2, 1])
    decode([3, 0, 2, 1], 8)
    # Release/reuse two slots at new lengths while retaining the other caches.
    del states[0], states[1]
    prefill([args.context-16, 65], [0, 1])
    decode([2, 0, 1, 3], 8)
    decode([0], 4)
    decode([3, 1], 3)
    window = model.config.get('sliding_window') if model.config['model_type'] in ('gemma3_text', 'phi3', 'mistral', 'starcoder2') else None
    if window and window + 8 < args.context:
        states.clear()
        prefill([window-1, window, window+1], [2, 0, 3])
        decode([3, 2, 0], 4)
    # An oversized prompt must not alter any length/cached state.
    if executor is not None:
        before = list(executor.lengths)
        try:
            executor.prefill_batch([[1] * (args.context+1)], [0])
            raise AssertionError('oversized prompt accepted')
        except ValueError:
            assert executor.lengths == before
    decode([next(iter(states))], 1)
    if model.config['model_type'] in ('gpt2','bloom'):
        prefill([args.context-1],[1])
        decode([1],1)
        if executor is not None:
            before=list(executor.lengths)
            try:
                executor.decode_batch([states[1][1]],[1])
                raise AssertionError('position past learned table accepted')
            except ValueError:
                assert executor.lengths==before
    if args.tiny_prefixes:
        # Exercise scalar lengths and packed totals on both sides of tiny
        # projection dispatch boundaries, with fresh and reused cache slots.
        tiny_cases = [[length] for length in range(1,18)] + [
            [1,1],[1,2],[1,1,1,1],[1,4],[2,2],[1,2,4],[3,5],[8,8],[5,11],[9,8]]
        for index,lengths in enumerate(tiny_cases):
            slots=[(3-index+offset)%4 for offset in range(len(lengths))]
            prefill(lengths,slots)
            decode(list(reversed(slots)),4)
    if args.golden_output:
        save_trace(args.golden_output, trace_metadata, trace, complete=True)
    print(json.dumps({'model_path': args.path, 'reference_dtype': args.reference_dtype, 'reference_attention': args.reference_attention, 'architecture': model.config['model_type'],
        'device': device_name(), 'torch': str(torch.__version__), 'reference_device': reference_device,
        'weight_dtype': str(executor.w['head'].dtype) if executor else None,
        'reference_only': args.reference_only, 'cpu_threads':args.cpu_threads,
        'packed_cpu_reference':args.packed_cpu_reference, 'packed_gpu_reference':args.packed_gpu_reference,
        'reference_quantization': args.reference_quantization,
        'weight_quantization': model.quantization if executor else None,
        'prefill_chunk_size':args.prefill_chunk_size, 'tiny_prefixes':args.tiny_prefixes,
        'context': args.context, 'backend': executor.kernel_backend if executor else None, 'checks': checks,
        'template_checks': template_checks, 'timings': timings,
        'passed': all(row['passed'] for row in checks) if executor else None}, indent=2))
    assert all(row['passed'] for row in checks), 'long-context correctness gate failed; see JSON checks'


if __name__ == '__main__':
    main()
