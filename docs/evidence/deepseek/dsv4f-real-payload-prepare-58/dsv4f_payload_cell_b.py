"""#58 payload cell B: full promoted/packed prepare of stage [0,22) — the
first host stage of the 2x96 GiB fleet plan — from the official weights."""
import gc, json, resource, sys, time
import torch
from damendaemon.deepseek_reference_loader import load_reference_stage

MODEL = "/mnt/nas-models/models/DeepSeek-V4-Flash-0731"

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

t0 = time.time()
stage = load_reference_stage(MODEL, 0, 22, max_weight_bytes=80 * 2**30)
dt = time.time() - t0
n = len(stage.blocks)
print(f"stage [0,22): loaded {n} blocks in {dt:.1f}s", flush=True)
print(f"RSS after load {rss_gib():.1f} GiB, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB", flush=True)
assert n == 22 and stage.embedding is not None and stage.head is None

# bounded integrity probes on the loaded stage: block-level structure across the range
ratios = [b.ratio for b in stage.blocks]
print("ratios:", ratios, flush=True)
experts = [len(b.packed_experts) for b in stage.blocks]
shared = [1 if b.packed_shared is not None else 0 for b in stage.blocks]
print("experts per block (set):", set(experts), "shared present:", sum(shared), "/22", flush=True)
wbytes = sum(p.weight.numel() + p.scales.numel() for b in stage.blocks for e in (*b.packed_experts, b.packed_shared) for p in (e.gate, e.up, e.down))
print(f"packed expert bytes held: {wbytes/2**30:.2f} GiB", flush=True)

# a single real forward probe: teacher-forced one-token prefill through block 0's
# reference machinery is executor work (next cell) — here only construction integrity.
stage.cleanup(); del stage; gc.collect()
print(f"after cleanup RSS {rss_gib():.1f} GiB", flush=True)
print("CELL-B-RESULT", json.dumps(dict(blocks=n, seconds=round(dt, 1), expert_bytes_gib=round(wbytes/2**30, 2),
      peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 1))), flush=True)
