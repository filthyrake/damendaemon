#!/usr/bin/env python3
"""#640 peak-memory probe: TokenGate Qwen3-1.7B NIAH-32K, full vs gated prefill.

Measures torch.cuda.max_memory_allocated / max_memory_reserved across each
arm's generate() call (prefill dominates: 32k prompt vs 32 decode tokens),
plus the model's own resident weights as the floor. Same samples/prompts as
examples/niah_demo.py.
"""
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HUB_REPO = "liana313/tokengate-niah-qwen3-1.7b"
SAMPLES = os.path.join(_REPO_ROOT, "examples", "data", "niah_32k_samples.jsonl")


def main():
    from tokengate import apply_chat_template, load_model
    from tokengate.datasets.tasks import get_task
    from transformers import AutoTokenizer
    import torch

    model = load_model(HUB_REPO, gate_attn="flex")
    tok = AutoTokenizer.from_pretrained(model.base_model)
    task = get_task("niah_kamradt")
    samples = [json.loads(l) for l in open(SAMPLES)]

    ids = [apply_chat_template(
               tok, task.format_prompt("{state}\n\nQuestion: " + s["question"]),
               state=s["body"])
           for s in samples]

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    def peak_for(k):
        # fresh peak tracking per arm; one warmup row first (CUDA init,
        # flex compile) so the measured rows are steady-state.
        model.generate([ids[0]], tok, max_kept_prefill_tokens=k)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for seq in ids:
            model.generate([seq], tok, max_kept_prefill_tokens=k)
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    print(f"weights resident: {weight_bytes / 2**30:.2f} GiB "
          f"(bf16 backbone + fp32 gate branch)")
    print(f"\n{'arm':>8} {'peak_alloc_GiB':>15} {'peak_res_GiB':>13}")
    print("-" * 40)
    out = {"weights_gib": weight_bytes / 2**30, "arms": {}}
    for label, k in [("full", None), ("2000", 2000), ("500", 500), ("100", 100)]:
        a, r = peak_for(k)
        out["arms"][label] = {"alloc_gib": a / 2**30, "reserved_gib": r / 2**30}
        print(f"{label:>8} {a / 2**30:>15.2f} {r / 2**30:>13.2f}")
    full_a = out["arms"]["full"]["alloc_gib"]
    for label in ("2000", "500", "100"):
        print(f"  ratio full/{label}: "
              f"{full_a / out['arms'][label]['alloc_gib']:.2f}x")
    json.dump(out, open(os.path.join(_REPO_ROOT, "tg-mem-probe.json"), "w"), indent=1)
    print("wrote tg-mem-probe.json")


if __name__ == "__main__":
    main()
