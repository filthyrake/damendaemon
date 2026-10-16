"""Independent-reference check of heterogeneous cache slots and batched decode."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM

from dllm.batch import BatchExecutor
from dllm.qwen2 import ModelWeights
from dllm.runtime import require_gpu, make_executor
from reference_model import load_reference
from dllm.device import selected_device, synchronize, empty_cache, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--reference-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument('--reference-device', choices=('cpu', 'cuda', 'mps'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference_device = args.reference_device or selected_device()
    require_gpu()
    model = ModelWeights(args.path, args.context)
    reference = load_reference(args.path, local_files_only=True,
        trust_remote_code=False, torch_dtype=getattr(torch, args.reference_dtype), attn_implementation="sdpa").to(reference_device).eval()
    executor = make_executor(model, max_batch=4)
    prompts = ["Explain why the sky looks blue in one sentence.",
               "What is 2 + 2? Answer with one number.",
               "Reply in French: Hello, how are you?"]
    slots = [3, 0, 2]  # Non-contiguous and deliberately out of order.
    expected, next_tokens = [], []
    errors = []
    with torch.inference_mode():
        for slot, prompt in zip(slots, prompts):
            ids = model.chat_tokens([{"role": "user", "content": prompt}])
            ref = reference(torch.tensor([ids], device=reference_device), use_cache=True)
            actual = executor.prefill(ids, slot)
            assert actual.argmax(-1).item() == ref.logits[:, -1].argmax(-1).item()
            expected.append(ref)
            next_tokens.append(ref.logits[:, -1].argmax(-1).item())
        # Re-prefill all slots together; packed GEMMs must preserve independent logits.
        packed = executor.prefill_batch([model.chat_tokens([{"role": "user", "content": prompt}])
                                        for prompt in prompts], slots)
        for row, ref in enumerate(expected):
            target = ref.logits[:, -1].float()
            error = (packed[row:row+1].float().to(target.device) - target).abs()
            assert error.square().mean().sqrt() / target.square().mean().sqrt() < .01
            assert error.max() < .5
            assert packed[row].argmax().item() == target.argmax().item()
        for step in range(16):
            actual, _ = executor.decode_batch(next_tokens, slots)
            for row, token in enumerate(next_tokens):
                expected[row] = reference(torch.tensor([[token]], device=reference_device),
                    past_key_values=expected[row].past_key_values, use_cache=True)
                target = expected[row].logits[:, -1].float()
                error = (actual[row:row+1].float().to(target.device) - target).abs()
                relative = (error.square().mean().sqrt() / target.square().mean().sqrt()).item()
                assert relative < 0.01 and error.max() < 0.5, (step, row, relative)
                assert actual[row].argmax().item() == target.argmax().item(), (step, row, "greedy mismatch")
                errors.append(relative)
                next_tokens[row] = target.argmax().item()
        # Reuse a released slot and change batch order/size without recapturing.
        ids = model.chat_tokens([{"role": "user", "content": "Name a primary color."}])
        token = executor.prefill(ids, 0).argmax(-1).item()
        ref = reference(torch.tensor([ids], device=reference_device), use_cache=True)
        ref = reference(torch.tensor([[token]], device=reference_device), past_key_values=ref.past_key_values, use_cache=True)
        actual, _ = executor.decode_batch([next_tokens[2], token], [2, 0])
        assert actual[1].argmax().item() == ref.logits[:, -1].argmax().item()
    del reference, expected, actual
    empty_cache()
    timings = []
    ids = model.chat_tokens([{"role": "user", "content": "Explain matrix multiplication in detail."}])
    for size in (1, 2, 4):
        for trial in range(3):
            tokens = executor.prefill_batch([ids] * size, list(range(size))).argmax(-1).tolist()
            start = time.perf_counter()
            for _ in range(63):
                _, next_ids = executor.decode_batch(tokens, list(range(size)))
                tokens = next_ids.tolist()
            elapsed = time.perf_counter() - start
            timings.append({"batch_size": size, "trial": trial, "seconds": elapsed,
                            "aggregate_decode_tokens_per_second": 63*size/elapsed})
    print(json.dumps({"reference_dtype": args.reference_dtype, "matched_batched_steps": len(errors), "slot_reuse_passed": True, "batched_prefill_passed": True,
                      "max_relative_rms_error": max(errors), "kernel_backend": executor.kernel_backend,
                      "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
