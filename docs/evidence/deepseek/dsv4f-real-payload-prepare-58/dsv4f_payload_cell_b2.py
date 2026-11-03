"""#58 payload cell B2: prepare of stage [22,43) — the second host stage of
the 2x96 GiB fleet plan (head boundary) — from the official weights."""
import gc, json, resource, sys, time
import torch
from damendaemon.deepseek_reference_loader import load_reference_stage

MODEL = "/mnt/nas-models/models/DeepSeek-V4-Flash-0731"
START, END = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (22, 43)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

t0 = time.time()
stage = load_reference_stage(MODEL, START, END, max_weight_bytes=80 * 2**30)
dt = time.time() - t0
n = len(stage.blocks)
print(f"stage [{START},{END}): loaded {n} blocks in {dt:.1f}s", flush=True)
print(f"RSS after load {rss_gib():.1f} GiB, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB", flush=True)
assert n == END - START and stage.embedding is None and stage.head is not None
ratios = [b.ratio for b in stage.blocks]
experts = {len(b.packed_experts) for b in stage.blocks}
shared = sum(1 if b.packed_shared is not None else 0 for b in stage.blocks)
print("ratios:", ratios, "experts:", experts, "shared:", shared, flush=True)
stage.cleanup(); del stage; gc.collect()
print(f"after cleanup RSS {rss_gib():.1f} GiB", flush=True)
print("CELL-B2-RESULT", json.dumps(dict(blocks=n, seconds=round(dt, 1),
      peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 1))), flush=True)
