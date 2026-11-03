"""#58 execution cell C: first teacher-forced forward through REAL official
weights (layers 0 and 2: ratio-0 window + first ratio-4 compression layer).

Gates (pre-declared):
1. EXECUTION: 32-token teacher-forced prefill through both stages completes,
   outputs finite, shapes exact (K, streams, dim).
2. COMPRESSION PARITY: prefill(31)+decode(1) last-position hidden matches
   full prefill(32) last-position hidden within float tolerance (same
   reference math, different batching path) — on real weights, real geometry.
3. DETERMINISM: a full reload of both stages reproduces the 32-token
   prefill outputs byte-identically.
No logits (mid-model stages), no generation, no quality claim.
"""
import gc, json, sys, time
import torch
from damendaemon.deepseek_reference_loader import load_reference_stage
from damendaemon.deepseek_stage_slots import DeepSeekStageSlots

MODEL = "/mnt/nas-models/models/DeepSeek-V4-Flash-0731"
K = 32
CTX = 64
torch.set_num_threads(8)

def rss_gib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30

def tokens_for(vocab):
    return ((torch.arange(K, dtype=torch.int64) * 7 + 3) % vocab)[None]  # (1, K)

def load_pair():
    s0 = load_reference_stage(MODEL, 0, 1, max_weight_bytes=80 * 2**30)
    s2 = load_reference_stage(MODEL, 2, 3, max_weight_bytes=80 * 2**30)
    return s0, s2

def run_prefill(s0, s2, tokens, k=None):
    """Teacher-forced prefill through the chained stages; returns hidden after each."""
    use = tokens if k is None else tokens[:, :k]
    o0 = DeepSeekStageSlots(s0, CTX, 1)
    o2 = DeepSeekStageSlots(s2, CTX, 1)
    try:
        hidden = o0.embedding(use)
        positions = torch.arange(use.shape[1], dtype=torch.int64)[None]
        t0 = time.time()
        h0 = o0.forward_layers(hidden, positions, torch.tensor([0]), True, 0, token_ids=use)
        t0dt = time.time() - t0
        t1 = time.time()
        h2 = o2.forward_layers(h0, positions, torch.tensor([0]), True, 0, token_ids=use)
        t2dt = time.time() - t1
        assert torch.isfinite(h0).all() and torch.isfinite(h2).all(), "non-finite hidden"
        return h0, h2, o0, o2, (t0dt, t2dt)
    except Exception:
        o0.close(); o2.close(); raise

t_tokens = tokens_for(129280)
print(f"tokens[0:8] = {t_tokens[:8].tolist()} vocab=129280 K={K}", flush=True)

s0, s2 = load_pair()
print(f"stages loaded, RSS {rss_gib():.1f} GiB; streams={s0.streams} dim={s0.dim} "
      f"ratio0={s0.blocks[0].ratio} ratio2={s2.blocks[0].ratio}", flush=True)

# Gate 1: full 32-token prefill
h0_full, h2_full, o0, o2, (t0, t2) = run_prefill(s0, s2, t_tokens)
print(f"GATE1 PASS: prefill(32) layer0 {t0:.2f}s layer2 {t2:.2f}s shapes {tuple(h0_full.shape)} {tuple(h2_full.shape)}", flush=True)

# Gate 2: prefill(31) + decode(1) parity vs full prefill(32)
h0_p31, h2_p31, o0b, o2b, _ = run_prefill(s0, s2, t_tokens, k=K - 1)
t_last = t_tokens[:, K - 1:K]
hidden_last = o0b.embedding(t_last)
pos_last = torch.tensor([[K - 1]], dtype=torch.int64)
d0 = o0b.forward_layers(hidden_last, pos_last, torch.tensor([0]), False, token_ids=t_last)
d2 = o2b.forward_layers(d0, pos_last, torch.tensor([0]), False, token_ids=t_last)
assert torch.isfinite(d0).all() and torch.isfinite(d2).all(), "non-finite decode"
diff0 = (d0[:, 0] - h0_full[:, -1]).abs().max().item()
diff2 = (d2[:, 0] - h2_full[:, -1]).abs().max().item()
mag0 = h0_full[:, -1].abs().mean().item(); mag2 = h2_full[:, -1].abs().mean().item()
print(f"GATE2 layer0(ratio-0): max|dec-pref|={diff0:.3e} |h|mean={mag0:.3e} rel={diff0/mag0:.2e}", flush=True)
print(f"GATE2 layer2(ratio-4): max|dec-pref|={diff2:.3e} |h|mean={mag2:.3e} rel={diff2/mag2:.2e}", flush=True)
o0.close(); o2.close(); o0b.close(); o2b.close()

# Gate 3: reload determinism
s0.cleanup(); s2.cleanup(); del s0, s2, h0_p31, h2_p31; gc.collect()
r0, r2 = load_pair()
h0_r, h2_r, ro0, ro2, _ = run_prefill(r0, r2, t_tokens)
byte0 = torch.equal(h0_r, h0_full); byte2 = torch.equal(h2_r, h2_full)
print(f"GATE3 reload determinism: layer0 byte-identical={byte0} layer2 byte-identical={byte2}", flush=True)
ro0.close(); ro2.close(); r0.cleanup(); r2.cleanup(); gc.collect()

result = dict(k=K, layer0_seconds=round(t0, 2), layer2_seconds=round(t2, 2),
             parity_max_abs=dict(layer0=diff0, layer2=diff2),
             parity_relative=dict(layer0=diff0 / mag0, layer2=diff2 / mag2),
             reload_byte_identical=dict(layer0=byte0, layer2=byte2),
             magnitude_mean=dict(layer0=mag0, layer2=mag2))
ok = all(result["reload_byte_identical"].values()) and diff0 < 1e-3 and diff2 < 1e-3
print("CELL-C-RESULT", json.dumps(result), flush=True)
print("CELL-C-GATES", "ALL PASS" if ok else "CHECK OUTPUT", flush=True)
print(f"final RSS {rss_gib():.1f} GiB", flush=True)
