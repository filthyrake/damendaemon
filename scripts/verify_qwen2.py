"""Real GPU correctness gate against a separately implemented HF reference.

Run with CUDA_VISIBLE_DEVICES=<one GPU UUID> python scripts/verify_qwen2.py PATH.
Transformers is a validation-only dependency, not the dllm inference runtime.
"""
import argparse
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dllm.qwen2 import ModelWeights, Executor
from dllm.runtime import require_gpu, make_executor
from reference_model import load_reference
from dllm.device import selected_device, synchronize, empty_cache, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--reference-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument('--reference-device', choices=('cpu', 'cuda', 'mps'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference_device = args.reference_device or selected_device()
    require_gpu()
    model = ModelWeights(args.path, args.context)
    tokenizer = AutoTokenizer.from_pretrained(args.path, local_files_only=True, trust_remote_code=False)
    reference = load_reference(
        args.path, local_files_only=True, trust_remote_code=False, torch_dtype=getattr(torch, args.reference_dtype),
        attn_implementation="sdpa").to(reference_device).eval()
    cases = []
    prompts = ["What is 2 + 2? Answer with one number.",
               "Explain why the sky looks blue in one sentence.",
               "Reply in French: Hello, how are you?"]
    for use_graph in (False, True):
        executor = make_executor(model, use_graph=use_graph, fuse_projections=not args.unfused)
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            tokens = model.chat_tokens(messages)
            assert tokens == tokenizer.apply_chat_template(messages, return_dict=False, add_generation_prompt=True), "chat template mismatch"
            inputs = torch.tensor([tokens], device=reference_device)
            with torch.inference_mode():
                expected = reference(inputs, use_cache=True)
                actual = executor.prefill(tokens)
                ids = []
                max_error, mean_error, relative_error = 0., 0., 0.
                mismatches = []
                for step in range(args.steps):
                    target = expected.logits[:, -1, :]
                    error = (actual.float().to(target.device) - target.float()).abs()
                    max_error = max(max_error, error.max().item())
                    mean_error = max(mean_error, error.mean().item())
                    relative = error.square().mean().sqrt() / target.float().square().mean().sqrt().clamp_min(1e-6)
                    relative_error = max(relative_error, relative.item())
                    token = target.argmax(-1).item()
                    if actual.argmax(-1).item() != token:
                        mismatches.append(step)
                    ids.append(token)
                    expected = reference(torch.tensor([[token]], device=reference_device),
                                         past_key_values=expected.past_key_values, use_cache=True)
                    actual, _ = executor.decode(token)
            cases.append({"graph": use_graph and selected_device() == "cuda", "graph_requested": use_graph, "prompt": prompt,
                          "matched_greedy_steps": len(ids) - len(mismatches), "greedy_mismatch_steps": mismatches,
                          "max_logit_error": max_error, "max_relative_rms_error": relative_error,
                          "max_mean_logit_error": mean_error,
                          "text": tokenizer.decode(ids, skip_special_tokens=True)})
        del executor
        gc.collect()
        empty_cache()
    del reference, expected, actual
    gc.collect()
    empty_cache()
    timings = []
    tokens = model.chat_tokens([{"role": "user", "content": "Explain matrix multiplication in detail."}])
    for use_graph in (False, True):
        executor = make_executor(model, use_graph=use_graph, fuse_projections=not args.unfused)
        for trial in range(3):
            synchronize()
            start = time.perf_counter()
            token = executor.prefill(tokens).argmax(-1).item()
            first = time.perf_counter()
            for _ in range(63):
                _, next_token = executor.decode(token)
                token = next_token.item()
            synchronize()
            end = time.perf_counter()
            timings.append({"graph": use_graph and selected_device() == "cuda", "graph_requested": use_graph, "trial": trial,
                            "first_token_seconds": first - start,
                            "decode_tokens_per_second": 63 / (end - first)})
        del executor
        gc.collect()
        empty_cache()
    print(json.dumps({"model_path": args.path, "reference_dtype": args.reference_dtype, "device": device_name(),
                      "torch": torch.__version__, "context": args.context,
                      "correctness": cases, "timings": timings}, indent=2))
    if not args.diagnostic:
        assert all(not c["greedy_mismatch_steps"] and c["max_relative_rms_error"] < 0.01
                   and c["max_logit_error"] < 0.5 for c in cases), "GPU correctness gate failed"


if __name__ == "__main__":
    main()
