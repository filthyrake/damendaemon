"""Profile CUDA decode kernels; timing comparisons belong in unprofiled runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from dllm.models import load
from dllm.runtime import require_gpu, make_executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument('--batch-size', type=int, choices=(1,2,4), default=1)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--output')
    parser.add_argument('--prompt-token-ids', help='JSON token IDs matching a serving benchmark')
    parser.add_argument('--experimental-awq-marlin', metavar='LIBRARY')
    args = parser.parse_args()
    require_gpu()
    experiment = None
    if args.experimental_awq_marlin:
        from experiments.awq_marlin_dispatch import install_validation_dispatch
        experiment = install_validation_dispatch(args.experimental_awq_marlin)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = load(args.path, args.context, args.quantization)
    host_seconds = time.perf_counter()-started
    started = time.perf_counter()
    executor = make_executor(model, use_graph=not args.eager, max_batch=args.batch_size)
    torch.cuda.synchronize()
    activation_seconds = time.perf_counter()-started
    prompt = (json.loads(Path(args.prompt_token_ids).read_text()) if args.prompt_token_ids else
              model.chat_tokens([{"role": "user", "content": "Explain matrix multiplication in detail."}]))
    if (not isinstance(prompt, list) or not prompt or
            any(type(token) is not int or token < 0 or token >= model.config['vocab_size'] for token in prompt)):
        raise ValueError('profile requires nonempty valid token IDs')
    if len(prompt)+3+args.steps > args.context:raise ValueError('profile exceeds context')
    slots = list(range(args.batch_size))
    token = (executor.prefill(prompt).argmax(-1).item() if args.batch_size==1 else
             executor.prefill_batch([prompt]*args.batch_size,slots).argmax(-1).tolist())
    def decode(token):
        if args.batch_size==1:
            _, next_token = executor.decode(token)
            return next_token.item()
        _, next_token = executor.decode_batch(token,slots)
        return next_token.tolist()
    for _ in range(3):token=decode(token)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(args.steps):
            token = decode(token)
        torch.cuda.synchronize()
    kernels = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]["calls"] += 1
            kernels[event.name]["total_us"] += event.self_device_time_total
    ordered = sorted(({"name": name, **value} for name, value in kernels.items()),
                     key=lambda value: value["total_us"], reverse=True)
    report = {"experiment":experiment, "batch_size":args.batch_size, "backend":executor.kernel_backend,
                      "host_preparation_seconds":host_seconds, "executor_activation_seconds":activation_seconds,
                      "context": args.context, "prompt_token_ids": prompt,
                      "steps": args.steps, "graph": not args.eager,
                      "kernel_calls_per_decode_step": sum(k["calls"] for k in ordered) / args.steps,
                      "kernel_time_us_per_decode_step": sum(k["total_us"] for k in ordered) / args.steps,
                      "kernels": ordered}
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
