"""#58 payload cell A: construct promoted/packed stages from the official
DSV4F-0731 weights; byte/value spot-checks vs independent safetensors reads."""
import gc, json, resource, sys, time
import torch
from safetensors import safe_open
from damendaemon.deepseek_reference_loader import load_reference_stage

MODEL = "/mnt/nas-models/models/DeepSeek-V4-Flash-0731"
WB = 80 * 2**30

index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

def direct(name):
    with safe_open(f"{MODEL}/{index[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def spot(stage, start, checks):
    ok = fail = 0
    for name, get, kind in checks:
        ours = get(stage)
        raw = direct(name)
        if kind == "fp32-from-bf16":
            same = torch.equal(ours, raw.to(torch.float32))
        elif kind == "i32":
            same = torch.equal(ours, raw.to(torch.int32))
        else:  # byte-exact uint8 (fp4/fp8 payload or scale)
            same = torch.equal(ours if ours.dtype == torch.uint8 else ours.view(torch.uint8),
                               raw if raw.dtype == torch.uint8 else raw.view(torch.uint8))
        print(f"  {'PASS' if same else 'FAIL'} {name} shape={tuple(ours.shape)} dtype={ours.dtype}")
        ok += same; fail += (not same)
    return ok, fail

results = {}
for start, end in ((0, 1), (22, 23), (42, 43)):
    t0 = time.time()
    stage = load_reference_stage(MODEL, start, end, max_weight_bytes=WB)
    dt = time.time() - t0
    n = len(stage.blocks)
    print(f"stage [{start},{end}): loaded {n} blocks in {dt:.1f}s, RSS {rss_gib():.1f} GiB, "
          f"embed={'yes' if stage.embedding is not None else 'no'}, head={'yes' if stage.head else 'no'}")
    checks = [
        (f"layers.{start}.attn.wq_a.weight", lambda s: s.blocks[0].attention_weights["wq_a.weight"].weight, "bytes"),
        (f"layers.{start}.attn.wq_a.scale", lambda s: s.blocks[0].attention_weights["wq_a.weight"].scales, "bytes"),
        (f"layers.{start}.ffn.experts.0.w1.weight", lambda s: s.blocks[0].packed_experts[0].gate.weight, "bytes"),
        (f"layers.{start}.ffn.experts.0.w1.scale", lambda s: s.blocks[0].packed_experts[0].gate.scales, "bytes"),
        (f"layers.{start}.ffn.experts.255.w2.weight", lambda s: s.blocks[0].packed_experts[255].down.weight, "bytes"),
        (f"layers.{start}.ffn.gate.weight", lambda s: s.blocks[0].weights["ffn.gate.weight"], "fp32-from-bf16"),
        (f"layers.{start}.attn_norm.weight", lambda s: s.blocks[0].weights["attn_norm.weight"], "fp32-from-bf16"),
    ]
    if "ffn.gate.tid2eid" in stage.blocks[0].weights:  # hash layers only (layers 0..2)
        checks.append((f"layers.{start}.ffn.gate.tid2eid", lambda s: s.blocks[0].weights["ffn.gate.tid2eid"], "i32"))
    if start == 0:
        checks.append(("embed.weight", lambda s: s.embedding, "fp32-from-bf16"))
    if end == 43:
        checks.append(("head.weight", lambda s: s.head["head.weight"], "fp32-from-bf16"))
        checks.append(("norm.weight", lambda s: s.head["norm.weight"], "fp32-from-bf16"))
    ok, fail = spot(stage, start, checks)
    results[f"[{start},{end})"] = dict(blocks=n, seconds=round(dt, 1), checks_passed=ok, checks_failed=fail)
    print(f"  spot-checks: {ok} passed, {fail} failed")
    stage.cleanup(); del stage; gc.collect()
    print(f"  after cleanup RSS {rss_gib():.1f} GiB")
    sys.stdout.flush()

print("CELL-A-RESULT", json.dumps(results))
print("peak RSS GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))
