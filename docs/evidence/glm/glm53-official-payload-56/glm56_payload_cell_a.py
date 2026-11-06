"""#56 payload cell A: stages from the staged official GLM-5.3 weights,
byte/value spot-checks vs independent safetensors reads. Covers the layer
kinds and the indexer-group discipline: dense + full-indexer MLA (layer 0),
a full sparse-MoE indexer group [6,10), and the final group [74,78) with
head/norm. Stage boundaries preserve full/shared indexer groups."""
import gc, json, resource, sys, time
import torch
from safetensors import safe_open
from damendaemon.glm_reference_loader import load_reference_stage

MODEL = "/mnt/nas-models/models/GLM-5.3"
WB = {(0, 1): 8 * 2**30, (6, 10): 48 * 2**30, (74, 78): 48 * 2**30}
PW = 64 * 2**20

index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

def direct(name):
    with safe_open(f"{MODEL}/{index[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def unwrap(v):
    return v.weight if hasattr(v, "weight") else v

def packed(name, get):
    ours = unwrap(get())
    raw = direct(name)
    ok = torch.equal(ours, raw.view(torch.uint8) if raw.dtype != torch.uint8 else raw)
    print(f"  {'PASS' if ok else 'FAIL'} [fp8-bytes] {name} shape={tuple(ours.shape)}")
    return ok

def scales(name, get):
    ours = get().scales if hasattr(get(), "scales") else unwrap(get())
    raw = direct(name)
    ok = torch.equal(ours.float(), raw.float())
    print(f"  {'PASS' if ok else 'FAIL'} [scales] {name} shape={tuple(ours.shape)}")
    return ok

def values(name, get):
    ours = unwrap(get())
    raw = direct(name)
    ok = torch.equal(ours.float(), raw.float())
    print(f"  {'PASS' if ok else 'FAIL'} [values] {name} shape={tuple(ours.shape)}")
    return ok

def q_weights(s, name):
    return s.blocks[0].attention.weights[name]

results = {}
TOTALS = {}
for start, end in ((0, 1), (6, 10), (74, 78)):
    t0 = time.time()
    stage = load_reference_stage(MODEL, start, end, max_weight_bytes=WB[(start, end)],
                                  projection_workspace_bytes=PW, max_workspace_bytes=PW * 2)
    dt = time.time() - t0
    b = stage.blocks[0]
    print(f"stage [{start},{end}): attn={type(b.attention).__name__} ffn={type(b.feed_forward).__name__} "
          f"blocks={len(stage.blocks)} in {dt:.1f}s, RSS {rss_gib():.1f} GiB, embed={'yes' if stage.embedding is not None else 'no'}, "
          f"head={'yes' if stage.head is not None else 'no'}")
    ok = 0
    if start == 0:
        ok += packed("model.layers.0.self_attn.q_a_proj.weight", lambda: q_weights(stage, "q_a_proj.weight"))
        ok += scales("model.layers.0.self_attn.q_a_proj.weight_scale_inv", lambda: q_weights(stage, "q_a_proj.weight"))
        ok += values("model.layers.0.self_attn.q_a_layernorm.weight", lambda: q_weights(stage, "q_a_layernorm.weight"))
        ok += packed("model.layers.0.self_attn.o_proj.weight", lambda: q_weights(stage, "o_proj.weight"))
        ok += scales("model.layers.0.self_attn.o_proj.weight_scale_inv", lambda: q_weights(stage, "o_proj.weight"))
        ok += packed("model.layers.0.self_attn.indexer.wk.weight", lambda: q_weights(stage, "indexer.wk.weight"))
        ok += values("model.layers.0.self_attn.indexer.k_norm.bias", lambda: q_weights(stage, "indexer.k_norm.bias"))
        ok += packed("model.layers.0.mlp.down_proj.weight", lambda: stage.blocks[0].feed_forward.experts[0].down)
        ok += scales("model.layers.0.mlp.down_proj.weight_scale_inv", lambda: stage.blocks[0].feed_forward.experts[0].down)
        ok += values("model.embed_tokens.weight", lambda: stage.embedding)
        TOTALS[(0, 1)] = 10
    if start == 6:
        ff = stage.blocks[0].feed_forward
        ok += packed("model.layers.6.self_attn.kv_a_proj_with_mqa.weight", lambda: q_weights(stage, "kv_a_proj_with_mqa.weight"))
        ok += packed("model.layers.6.self_attn.q_b_proj.weight", lambda: q_weights(stage, "q_b_proj.weight"))
        ok += packed("model.layers.6.self_attn.kv_b_proj.weight", lambda: q_weights(stage, "kv_b_proj.weight"))
        ok += packed("model.layers.6.mlp.experts.0.gate_proj.weight", lambda: ff.experts[0].gate)
        ok += scales("model.layers.6.mlp.experts.0.gate_proj.weight_scale_inv", lambda: ff.experts[0].gate)
        ok += packed("model.layers.6.mlp.experts.255.down_proj.weight", lambda: ff.experts[255].down)
        ok += packed("model.layers.6.mlp.shared_experts.up_proj.weight", lambda: ff.shared_experts.gate if hasattr(ff, "shared_experts") else ff.shared.gate)
        ok += values("model.layers.6.mlp.gate.weight", lambda: ff.router)
        ok += values("model.layers.6.mlp.gate.e_score_correction_bias", lambda: ff.correction_bias)
        ok += packed("model.layers.9.mlp.experts.7.up_proj.weight", lambda: stage.blocks[3].feed_forward.experts[7].up)
        TOTALS[(6, 10)] = 10
    if start == 74:
        ff = stage.blocks[3].feed_forward  # layer 77
        ok += packed("model.layers.77.self_attn.kv_b_proj.weight", lambda: stage.blocks[3].attention.weights["kv_b_proj.weight"])
        ok += packed("model.layers.77.mlp.experts.7.up_proj.weight", lambda: ff.experts[7].up)
        ok += values("model.layers.77.mlp.gate.weight", lambda: ff.router)
        ok += values("lm_head.weight", lambda: stage.head)
        ok += values("model.norm.weight", lambda: stage.norm)
        TOTALS[(74, 78)] = 5
    print(f"  spot-checks: {ok}/{TOTALS[(start, end)]} passed")
    results[f"[{start},{end})"] = dict(seconds=round(dt, 1), checks_passed=ok, checks_total=TOTALS[(start, end)])
    stage.cleanup(); del stage; gc.collect()
    print(f"  after cleanup RSS {rss_gib():.1f} GiB")
    sys.stdout.flush()

print("GLM53-CELL-A-RESULT", json.dumps(results))
print("peak RSS GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))
