# GLM-5.3 q4 fleet execution — rung 4b: q4 attention projections (#719)

2026-11-08/09, rocky10-desktop A6000, guarded (thermal 90 / reserve 0 /
1500 s, allocator 12 GiB). Result: **HONEST NEGATIVE — pre-declared gates
FAIL; attention stays shipped-fp8 in the fleet chain.**

## Architecture finding (attempt 1's scope error, retained)

GLM-5.3 is `model_type: glm_moe_dsa` (GlmMoeDsaForCausalLM) with a FLAT
config — not the glm5_next "flash" family. The production path is
`CudaCachedGlmAttention` (MLA projection → full indexer selection → sparse
latent attention → kv_b + output projection). Attempt 1 refused at 1.3 s on
a `text_config` KeyError from aiming at the flash modules; retained as
`evidence-attempt1.json` / `guard-attempt1.json`.

## The measurement (attempts 2/3, identical numbers — deterministic)

Real layer-6 weights (full-indexer layer), production
`CudaCachedGlmAttention`, oracle = shipped fp8 projections vs candidate =
the same five projections (`q_a`, `q_b`, `kv_a_with_mqa`, `kv_b`, `o`)
packed to canonical q4_0 through the widened admission. Indexer
(`wq_b`/`wk`) shipped fp8 in BOTH runs; norms/indexer-dense shared by
reference. 64 tokens, seed 719. Guard clean, ~30 s, peak reserved 317 MiB.

- **attention_parity: rel RMS 0.1611 > 0.15** (max_abs 0.107×absmax passed).
  Weight-level drift is the standing 8.7–10.2% family; the divergence
  compounds through selection.
- **selection_exactness: FAIL — 976/4096 selection entries (23.8%) differ,
  56/64 queries affected.** Root cause is architectural: the DSA indexer's
  query derives from the MLA low-rank (`select(hidden, projected['low_rank'],
  ...)`), so quantizing `q_a_proj` perturbs the low-rank and with it the
  token selection. Selection-exactness (#668) is NOT independent of
  attention-projection quantization in this architecture.
- finiteness: PASS. memory: q4 92.8 MB vs fp8 165.0 MB (PASS).

## Lane decision for #719

The fleet chain keeps **attention + indexer in shipped fp8** — selection
exactness wins (#668), and the storage cost is negligible (165 MB/layer
fp8 ≈ 12.9 GiB across 78 layers vs the ~309 GiB expert banks). The q4
surface stays where rungs 2–4a proved it: MoE expert banks + shared
experts, through the qualified executor, streaming per rung 4a.

The admission widening in `glm_attention._projection` merges anyway: it is
correct, CI-tested, documents the boundary this cell measured, and keeps
the packed-projection surface uniform with the FFN bank admission (#721).
