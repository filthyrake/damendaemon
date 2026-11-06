"""#56 execution cell C: teacher-forced forward through REAL GLM-5.3 weights
(layer 0: dense MLP + full-indexer MLA; layer 6: sparse top-8/256 MoE +
group-head MLA) — the two boundary kinds of this family.

Gates (pre-declared, #705 pattern):
1. EXECUTION: 32-token prefill through both stages completes; finite; shapes exact.
2. STATE PARITY: prefill(31)+decode(1) last-position hidden matches full
   prefill(32) last-position hidden — dense/indexer-full path AND the
   sparse-MoE + KV-cache path at real geometry.
3. RELOAD DETERMINISM: full reload reproduces prefill outputs byte-identically.
No logits (mid-model), no generation, no quality claim.
"""
import gc, json, sys, time
import torch
from damendaemon.glm_reference_loader import load_reference_stage
from damendaemon.glm_stage_slots import GlmStageSlots

MODEL = "/mnt/nas-models/models/GLM-5.3"
K = 32
CTX = 64
WB0, WB6 = 8 * 2**30, 48 * 2**30
PW = 64 * 2**20
torch.set_num_threads(8)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def tokens_for(vocab):
    return ((torch.arange(K, dtype=torch.int64) * 7 + 3) % vocab)[None]

def load_pair():
    s0 = load_reference_stage(MODEL, 0, 1, max_weight_bytes=WB0,
                              projection_workspace_bytes=PW, max_workspace_bytes=PW * 2)
    s6 = load_reference_stage(MODEL, 6, 10, max_weight_bytes=WB6,
                              projection_workspace_bytes=PW, max_workspace_bytes=PW * 2)
    return s0, s6

def run_prefill(s0, s6, tokens, k=None):
    use = tokens if k is None else tokens[:, :k]
    o0 = GlmStageSlots(s0, CTX, 1)
    o6 = GlmStageSlots(s6, CTX, 1)
    try:
        hidden = o0.embedding(use)
        positions = torch.arange(use.shape[1], dtype=torch.int64)[None]
        t0 = time.time()
        h0 = o0.forward_layers(hidden, positions, torch.tensor([0]), True, 0)
        t0dt = time.time() - t0
        t1 = time.time()
        h6 = o6.forward_layers(h0, positions, torch.tensor([0]), True, 0)
        t6dt = time.time() - t1
        assert torch.isfinite(h0).all() and torch.isfinite(h6).all(), "non-finite hidden"
        return h0, h6, o0, o6, (t0dt, t6dt)
    except Exception:
        o0.close(); o6.close(); raise

t_tokens = tokens_for(154880)
print(f"tokens[0:8] = {t_tokens[0,:8].tolist()} vocab=154880 K={K}", flush=True)

s0, s6 = load_pair()
print(f"stages loaded, RSS {rss_gib():.1f} GiB; dim={s0.dim} "
      f"layer0={type(s0.blocks[0].attention).__name__}/{type(s0.blocks[0].feed_forward).__name__} "
      f"layer6={type(s6.blocks[0].attention).__name__}/{type(s6.blocks[0].feed_forward).__name__}", flush=True)

h0_full, h6_full, o0, o6, (t0, t6) = run_prefill(s0, s6, t_tokens)
print(f"GATE1 PASS: prefill(32) layer0 {t0:.2f}s layer6 {t6:.2f}s shapes {tuple(h0_full.shape)} {tuple(h6_full.shape)}", flush=True)

h0_p31, h6_p31, o0b, o6b, _ = run_prefill(s0, s6, t_tokens, k=K - 1)
t_last = t_tokens[:, K - 1:K]
hidden_last = o0b.embedding(t_last)
pos_last = torch.tensor([[K - 1]], dtype=torch.int64)
d0 = o0b.forward_layers(hidden_last, pos_last, torch.tensor([0]), False)
d6 = o6b.forward_layers(d0, pos_last, torch.tensor([0]), False)
assert torch.isfinite(d0).all() and torch.isfinite(d6).all(), "non-finite decode"
diff0 = (d0[:, 0] - h0_full[:, -1]).abs().max().item()
diff6 = (d6[:, 0] - h6_full[:, -1]).abs().max().item()
mag0 = h0_full[:, -1].abs().mean().item(); mag6 = h6_full[:, -1].abs().mean().item()
print(f"GATE2 layer0(dense+full-idx MLA): max|dec-pref|={diff0:.3e} |h|mean={mag0:.3e} rel={diff0/mag0:.2e}", flush=True)
print(f"GATE2 layer6(sparse top-8/256 MoE): max|dec-pref|={diff6:.3e} |h|mean={mag6:.3e} rel={diff6/mag6:.2e}", flush=True)
o0.close(); o6.close(); o0b.close(); o6b.close()

s0.cleanup(); s6.cleanup(); del s0, s6, h0_p31, h6_p31; gc.collect()
r0, r6 = load_pair()
h0_r, h6_r, ro0, ro6, _ = run_prefill(r0, r6, t_tokens)
byte0 = torch.equal(h0_r, h0_full); byte6 = torch.equal(h6_r, h6_full)
print(f"GATE3 reload determinism: layer0 byte-identical={byte0} layer6 byte-identical={byte6}", flush=True)
ro0.close(); ro6.close(); r0.cleanup(); r6.cleanup(); gc.collect()

result = dict(k=K, layer0_seconds=round(t0, 2), layer6_seconds=round(t6, 2),
             parity_max_abs=dict(layer0=diff0, layer6=diff6),
             parity_relative=dict(layer0=diff0 / mag0, layer6=diff6 / mag6),
             reload_byte_identical=dict(layer0=byte0, layer6=byte6))
ok = byte0 and byte6 and diff0 < 1e-3 and diff6 < 1e-3
print("GLM53-CELL-C-RESULT", json.dumps(result), flush=True)
print("GLM53-CELL-C-GATES", "ALL PASS" if ok else "CHECK OUTPUT", flush=True)
print(f"final RSS {rss_gib():.1f} GiB", flush=True)
