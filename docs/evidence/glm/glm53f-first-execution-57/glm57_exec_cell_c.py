"""#57 execution cell C: teacher-forced forward through REAL GLM-5.3-Flash
weights (layer 0: gated-deltanet linear attention; layer 3: sparse MLA +
indexer) — the two attention boundary kinds of this family.

Gates (pre-declared, #705 pattern):
1. EXECUTION: 32-token prefill through both stages completes; finite; shapes exact.
2. STATE PARITY: prefill(31)+decode(1) last-position hidden matches full
   prefill(32) last-position hidden — recurrent-state path (linear) AND
   sparse-indexer/KV path (MLA) at real geometry.
3. RELOAD DETERMINISM: full reload reproduces prefill outputs byte-identically.
No logits (mid-model), no generation, no quality claim.
"""
import gc, json, sys, time
import torch
from damendaemon.glm_flash_reference_loader import load_reference_stage
from damendaemon.glm_flash_stage_slots import FlashStageSlots

MODEL = "/mnt/nas-models/models/GLM-5.3-Flash"
K = 32
CTX = 64
WB = 80 * 2**30
WS = 2 * 1024**3  # real KDA geometry needs ~17 MB/token; 32-token prefill ~832 MiB
torch.set_num_threads(8)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def tokens_for(vocab):
    return ((torch.arange(K, dtype=torch.int64) * 7 + 3) % vocab)[None]

def load_pair():
    s0 = load_reference_stage(MODEL, 0, 1, max_weight_bytes=WB, max_workspace_bytes=WS)
    s3 = load_reference_stage(MODEL, 3, 4, max_weight_bytes=WB, max_workspace_bytes=WS)
    return s0, s3

def run_prefill(s0, s3, tokens, k=None):
    use = tokens if k is None else tokens[:, :k]
    o0 = FlashStageSlots(s0, CTX, 1)
    o3 = FlashStageSlots(s3, CTX, 1)
    try:
        hidden = o0.embedding(use)
        positions = torch.arange(use.shape[1], dtype=torch.int64)[None]
        t0 = time.time()
        h0 = o0.forward_layers(hidden, positions, torch.tensor([0]), True, 0)
        t0dt = time.time() - t0
        t1 = time.time()
        h3 = o3.forward_layers(h0, positions, torch.tensor([0]), True, 0)
        t3dt = time.time() - t1
        assert torch.isfinite(h0).all() and torch.isfinite(h3).all(), "non-finite hidden"
        return h0, h3, o0, o3, (t0dt, t3dt)
    except Exception:
        o0.close(); o3.close(); raise

t_tokens = tokens_for(154880)
print(f"tokens[0:8] = {t_tokens[0,:8].tolist()} vocab=154880 K={K}", flush=True)

s0, s3 = load_pair()
print(f"stages loaded, RSS {rss_gib():.1f} GiB; streams={s0.streams} dim={s0.dim} "
      f"layer0={type(s0.blocks[0].attention).__name__} layer3={type(s3.blocks[0].attention).__name__}", flush=True)

h0_full, h3_full, o0, o3, (t0, t3) = run_prefill(s0, s3, t_tokens)
print(f"GATE1 PASS: prefill(32) layer0 {t0:.2f}s layer3 {t3:.2f}s shapes {tuple(h0_full.shape)} {tuple(h3_full.shape)}", flush=True)

h0_p31, h3_p31, o0b, o3b, _ = run_prefill(s0, s3, t_tokens, k=K - 1)
t_last = t_tokens[:, K - 1:K]
hidden_last = o0b.embedding(t_last)
pos_last = torch.tensor([[K - 1]], dtype=torch.int64)
d0 = o0b.forward_layers(hidden_last, pos_last, torch.tensor([0]), False)
d3 = o3b.forward_layers(d0, pos_last, torch.tensor([0]), False)
assert torch.isfinite(d0).all() and torch.isfinite(d3).all(), "non-finite decode"
diff0 = (d0[:, 0] - h0_full[:, -1]).abs().max().item()
diff3 = (d3[:, 0] - h3_full[:, -1]).abs().max().item()
mag0 = h0_full[:, -1].abs().mean().item(); mag3 = h3_full[:, -1].abs().mean().item()
print(f"GATE2 layer0(linear/gdn): max|dec-pref|={diff0:.3e} |h|mean={mag0:.3e} rel={diff0/mag0:.2e}", flush=True)
print(f"GATE2 layer3(sparse MLA+idx): max|dec-pref|={diff3:.3e} |h|mean={mag3:.3e} rel={diff3/mag3:.2e}", flush=True)
o0.close(); o3.close(); o0b.close(); o3b.close()

s0.cleanup(); s3.cleanup(); del s0, s3, h0_p31, h3_p31; gc.collect()
r0, r3 = load_pair()
h0_r, h3_r, ro0, ro3, _ = run_prefill(r0, r3, t_tokens)
byte0 = torch.equal(h0_r, h0_full); byte3 = torch.equal(h3_r, h3_full)
print(f"GATE3 reload determinism: layer0 byte-identical={byte0} layer3 byte-identical={byte3}", flush=True)
ro0.close(); ro3.close(); r0.cleanup(); r3.cleanup(); gc.collect()

result = dict(k=K, layer0_seconds=round(t0, 2), layer3_seconds=round(t3, 2),
             parity_max_abs=dict(layer0=diff0, layer3=diff3),
             parity_relative=dict(layer0=diff0 / mag0, layer3=diff3 / mag3),
             reload_byte_identical=dict(layer0=byte0, layer3=byte3))
ok = byte0 and byte3 and diff0 < 1e-3 and diff3 < 1e-3
print("GLM-CELL-C-RESULT", json.dumps(result), flush=True)
print("GLM-CELL-C-GATES", "ALL PASS" if ok else "CHECK OUTPUT", flush=True)
print(f"final RSS {rss_gib():.1f} GiB", flush=True)
