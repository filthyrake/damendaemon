"""Long-prefix, split-boundary and rolling-slot checks against an independent decoder."""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm.batch import BatchExecutor
from dllm.qwen2 import ModelWeights
from dllm.runtime import require_gpu, make_executor
from reference_model import load_reference
from dllm.device import selected_device, synchronize, empty_cache, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument("--reference-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument('--golden-output', help='save teacher-forced inputs and FP32 reference logits for cross-environment replay')
    parser.add_argument('--reference-attention', choices=('sdpa', 'eager'), default='sdpa')
    parser.add_argument('--reference-device', choices=('cpu', 'cuda', 'mps'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference_device = args.reference_device or selected_device()
    if args.context < 1040:
        parser.error("context must be at least 1040 for split-boundary tests")
    require_gpu()
    model = ModelWeights(args.path, args.context)
    tokenizer = AutoTokenizer.from_pretrained(args.path, local_files_only=True, trust_remote_code=False)
    reference = load_reference(args.path, local_files_only=True,
        trust_remote_code=False, torch_dtype=getattr(torch, args.reference_dtype), attn_implementation=args.reference_attention).to(reference_device).eval()
    executor = make_executor(model, max_batch=4)
    checks, template_checks, states, timings = [], [], {}, []
    trace = []
    records = 200
    while True:
        documents = ['Record %d: The parcel is stored in cabinet %d, drawer %d.\n' %
                     (i, i % 11, i % 7) for i in range(records)]
        corpus = tokenizer.encode(''.join(documents), add_special_tokens=False)
        if len(corpus) >= args.context + 64:
            break
        records *= 2

    def compare(actual, target, phase, slot):
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
        prompts = [corpus[slot*13:slot*13+length] for slot, length in zip(slots, lengths)]
        assert all(len(ids) == length for ids, length in zip(prompts, lengths))
        synchronize()
        start = time.perf_counter()
        actual = executor.prefill_batch(prompts, slots)
        synchronize()
        timings.append({'kind': 'batched_prefill', 'lengths': lengths, 'seconds': time.perf_counter()-start})
        event = {'kind': 'prefill', 'prompts': prompts, 'slots': slots, 'targets': []}
        trace.append(event)
        for row, (slot, ids) in enumerate(zip(slots, prompts)):
            ref = reference(torch.tensor([ids], device=reference_device), use_cache=True, logits_to_keep=1)
            target = ref.logits[:, -1]
            event['targets'].append(target.detach().float().cpu())
            compare(actual[row], target, 'prefill:'+str(len(ids)), slot)
            states[slot] = (ref.past_key_values, target.argmax().item())

    @torch.inference_mode()
    def decode(slots, steps):
        for step in range(steps):
            tokens = [states[slot][1] for slot in slots]
            actual, _ = executor.decode_batch(tokens, slots)
            event = {'kind': 'decode', 'tokens': tokens, 'slots': slots, 'targets': []}
            trace.append(event)
            for row, (slot, token) in enumerate(zip(slots, tokens)):
                ref = reference(torch.tensor([[token]], device=reference_device),
                    past_key_values=states[slot][0], use_cache=True, logits_to_keep=1)
                target = ref.logits[:, -1]
                event['targets'].append(target.detach().float().cpu())
                compare(actual[row], target, f'decode:{executor.lengths[slot]}', slot)
                states[slot] = (ref.past_key_values, target.argmax().item())

    # Real templates, including Unicode, assistant turns, and Qwen3 reasoning controls.
    modes = (None, False, True) if model.config['model_type'] in ('qwen3', 'qwen3_5_text') else (None,)
    messages = [{'role': 'system', 'content': 'Answer briefly.'},
                {'role': 'user', 'content': 'こんにちは — café 🦙. What is 7 × 8?'},
                {'role': 'assistant', 'content': '56.'},
                {'role': 'user', 'content': 'Return it as JSON.'}]
    for mode in modes:
        options = {} if mode is None else {'enable_thinking': mode}
        ids = model.chat_tokens(messages, **options)
        assert ids == tokenizer.apply_chat_template(messages, return_dict=False, add_generation_prompt=True, **options)
        with torch.inference_mode():
            actual = executor.prefill(ids, 0)
            ref = reference(torch.tensor([ids], device=reference_device), use_cache=True, logits_to_keep=1)
            compare(actual, ref.logits[:, -1], f'template:{mode}', 0)
            trace.append({'kind': 'prefill', 'prompts': [ids], 'slots': [0],
                          'targets': [ref.logits[:, -1].detach().float().cpu()]})
        template_checks.append({'enable_thinking': mode, 'tokens': len(ids)})
    del ref
    prefill([31, 255, 257, 1023], [3, 0, 2, 1])
    decode([3, 0, 2, 1], 8)
    # Release/reuse two slots at new lengths while retaining the other caches.
    del states[0], states[1]
    prefill([args.context-16, 65], [0, 1])
    decode([2, 0, 1, 3], 8)
    decode([0], 4)
    decode([3, 1], 3)
    if model.config['model_type'] == 'gemma3_text':
        states.clear()
        prefill([511, 512, 513], [2, 0, 3])
        decode([3, 2, 0], 4)
    # An oversized prompt must not alter any length/cached state.
    before = list(executor.lengths)
    try:
        executor.prefill_batch([[1] * (args.context+1)], [0])
        raise AssertionError('oversized prompt accepted')
    except ValueError:
        assert executor.lengths == before
    decode([next(iter(states))], 1)
    if args.golden_output:
        torch.save({'path': args.path, 'context': args.context, 'reference_dtype': args.reference_dtype,
                    'reference_torch': str(torch.__version__), 'reference_attention': args.reference_attention, 'trace': trace}, args.golden_output)
    print(json.dumps({'model_path': args.path, 'reference_dtype': args.reference_dtype, 'reference_attention': args.reference_attention, 'architecture': model.config['model_type'],
        'device': device_name(), 'torch': str(torch.__version__), 'reference_device': reference_device,
        'weight_dtype': str(executor.w['head'].dtype),
        'context': args.context, 'backend': executor.kernel_backend, 'checks': checks,
        'template_checks': template_checks, 'timings': timings,
        'passed': all(row['passed'] for row in checks)}, indent=2))
    assert all(row['passed'] for row in checks), 'long-context correctness gate failed; see JSON checks'


if __name__ == '__main__':
    main()
