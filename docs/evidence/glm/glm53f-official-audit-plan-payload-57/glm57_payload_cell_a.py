"""#57 payload cell A: single-layer stages from official GLM-5.3-Flash weights,
byte/value spot-checks vs independent safetensors reads. Covers all three
block kinds: linear-attention+dense FFN (layer 0), sparse MLA+MoE (layer 3),
last main layer + head (layer 44)."""
import gc, json, resource, sys, time
import torch
from safetensors import safe_open
from damendaemon.glm_flash_reference_loader import load_reference_stage

MODEL = "/mnt/nas-models/models/GLM-5.3-Flash"
ROOT = "model.language_model."
WB = 80 * 2**30

index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

def direct(name):
    with safe_open(f"{MODEL}/{index[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def unwrap(v):
    return v.weight if hasattr(v, "weight") else v

def spot(stage, checks):
    ok = fail = 0
    for name, get in checks:
        ours = get(stage)
        raw = direct(name)
        if ours.dtype == torch.uint8:
            same = torch.equal(ours, raw.view(torch.uint8) if raw.dtype != torch.uint8 else raw)
            kind = "bytes"
        else:
            same = torch.equal(ours.float(), raw.float())
            kind = f"values({str(ours.dtype)[6:]})"
        print(f"  {'PASS' if same else 'FAIL'} [{kind}] {name} shape={tuple(ours.shape)}")
        ok += same; fail += (not same)
    return ok, fail

results = {}
for start, end in ((0, 1), (3, 4), (44, 45)):
    t0 = time.time()
    stage = load_reference_stage(MODEL, start, end, max_weight_bytes=WB)
    dt = time.time() - t0
    b = stage.blocks[0]
    print(f"stage [{start},{end}): {type(b.attention).__name__}/{type(b.feed_forward).__name__} "
          f"in {dt:.1f}s, RSS {rss_gib():.1f} GiB, embed={'yes' if stage.embedding is not None else 'no'}, "
          f"head={'yes' if stage.head is not None else 'no'}")
    checks = []
    if start == 0:
        checks += [
            (ROOT + "layers.0.self_attn.q_proj.weight", lambda s: s.blocks[0].attention.weights["q_proj.weight"]),
            (ROOT + "layers.0.self_attn.o_proj.weight", lambda s: s.blocks[0].attention.weights["o_proj.weight"]),
            (ROOT + "layers.0.self_attn.A_log", lambda s: s.blocks[0].attention.weights["A_log"]),
            (ROOT + "layers.0.self_attn.dt_bias", lambda s: s.blocks[0].attention.weights["dt_bias"]),
            (ROOT + "embed_tokens.weight", lambda s: s.embedding),
        ]
    if start == 3:
        at3 = lambda s: s.blocks[0].attention
        ix = lambda s: s.blocks[0].attention.indexer
        ff3 = lambda s: s.blocks[0].feed_forward
        checks += [
            (ROOT + "layers.3.self_attn.kv_a_proj_with_mqa.weight", lambda s: at3(s).weights["kv_a_proj_with_mqa.weight"].weight),
            (ROOT + "layers.3.self_attn.kv_a_proj_with_mqa.weight_scale_inv", lambda s: at3(s).weights["kv_a_proj_with_mqa.weight"].scales),
            (ROOT + "layers.3.self_attn.kv_b_proj.weight", lambda s: unwrap(at3(s).weights["kv_b_proj.weight"])),
            (ROOT + "layers.3.self_attn.indexer.wk.weight", lambda s: unwrap(ix(s).weights["wk.weight"]) if hasattr(ix(s), "weights") else unwrap(ix(s).wk)),
            (ROOT + "layers.3.mlp.experts.0.gate_proj.weight", lambda s: ff3(s).experts[0].gate.weight),
            (ROOT + "layers.3.mlp.experts.0.gate_proj.weight_scale_inv", lambda s: ff3(s).experts[0].gate.scales),
            (ROOT + "layers.3.mlp.experts.287.down_proj.weight", lambda s: ff3(s).experts[287].down.weight),
            (ROOT + "layers.3.mlp.gate.weight", lambda s: unwrap(ff3(s).router)),
            (ROOT + "layers.3.mlp.gate.e_score_correction_bias", lambda s: unwrap(ff3(s).correction_bias)),
        ]
    if start == 44:
        checks += [
            (ROOT + "layers.44.self_attn.q_proj.weight", lambda s: s.blocks[0].attention.weights["q_proj.weight"]),
            (ROOT + "layers.44.mlp.experts.5.up_proj.weight", lambda s: s.blocks[0].feed_forward.experts[5].up.weight),
            (ROOT + "layers.44.mlp.gate.weight", lambda s: unwrap(s.blocks[0].feed_forward.router)),
        ]

    ok, fail = spot(stage, checks)
    if stage.head is not None:
        for name in ("lm_head.weight", ROOT + "norm.weight"):
            if name in index:
                raw = direct(name)
                # the final stage carries the unquantized head/norm as tensors
                value = stage.head if name == "lm_head.weight" else None
                if value is not None and getattr(value, "ndim", 0) == 2:
                    same = torch.equal(value.float(), raw.float())
                    print(f"  {'PASS' if same else 'FAIL'} [head] {name} shape={tuple(value.shape)}")
                    ok += same; fail += (not same)
    results[f"[{start},{end})"] = dict(seconds=round(dt, 1), checks_passed=ok, checks_failed=fail)
    print(f"  spot-checks: {ok} passed, {fail} failed")
    stage.cleanup(); del stage; gc.collect()
    print(f"  after cleanup RSS {rss_gib():.1f} GiB")
    sys.stdout.flush()

print("GLM-CELL-A-RESULT", json.dumps(results))
print("peak RSS GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))
