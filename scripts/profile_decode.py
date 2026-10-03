"""Profile CUDA decode kernels; timing comparisons belong in unprofiled runs."""
import argparse
from collections import defaultdict
import json

import torch

from dllm.qwen2 import ModelWeights, Executor
from dllm.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    require_gpu()
    model = ModelWeights(args.path, args.context)
    executor = Executor(model, use_graph=not args.eager)
    prompt = model.chat_tokens([{"role": "user", "content": "Explain matrix multiplication in detail."}])
    token = executor.prefill(prompt).argmax(-1).item()
    for _ in range(3):
        _, next_token = executor.decode(token)
        token = next_token.item()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(args.steps):
            _, next_token = executor.decode(token)
            token = next_token.item()
        torch.cuda.synchronize()
    kernels = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]["calls"] += 1
            kernels[event.name]["total_us"] += event.self_device_time_total
    ordered = sorted(({"name": name, **value} for name, value in kernels.items()),
                     key=lambda value: value["total_us"], reverse=True)
    print(json.dumps({"context": args.context, "steps": args.steps, "graph": not args.eager,
                      "kernel_calls_per_token": sum(k["calls"] for k in ordered) / args.steps,
                      "kernel_time_us_per_token": sum(k["total_us"] for k in ordered) / args.steps,
                      "kernels": ordered}, indent=2))


if __name__ == "__main__":
    main()
