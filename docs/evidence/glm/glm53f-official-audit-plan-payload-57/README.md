# #57 — GLM-5.3-Flash official weights: audit, fleet plan, payload prepare — zero code change

**First physical-qualification increment for #57** — and unlike the DSV4F
(#58) ladder, every stage of this one ran on the repo AS-MERGED through
`b2756f1b` with **no code change**: the flash contract admits the official
release directly, and the #703/#707 real-scale amendments (planner budgets,
transport/TLS deadlines, builder bounds, identity forwarding) carried this
family for free.

## 1. Contract + storage audit (PASS, as-merged)

- Official release `zai-org/GLM-5.3-Flash` (74 files, 328,366,173,639 bytes,
  62 shards — byte-verified vs the HF listing at download time) at
  `/mnt/nas-models/models/GLM-5.3-Flash`, NAS read-only.
- `glm_flash_checkpoint.tensor_specs` accepts the official config untouched:
  **74,001 main tensors** predicted. `inspect_reference_storage(family='flash')`
  passes the full header accounting against loader admission: retained
  **309.49 GiB** across 45 layers (per-layer 0.66–7.3 GiB — 3 dense-MLP
  lead-in layers vs 288-expert MoE layers), embedding 2.36 GiB, head 2.36
  GiB, largest conversion overlap 1.18 GiB.
- Structure decoded: 34 gated-deltanet linear-attention layers + 11 sparse
  MLA+indexer layers (every 4th from 3), `first_k_dense_replace` 3,
  288 routed + 1 shared expert, top-8 sigmoid noaux_tc routing, fp8-e4m3
  with 128×128 block scales (`weight_scale_inv` naming), vocab 154,880,
  hc_mult 4, 1M max positions. The MTP layer (`layers.45`: eh_proj/enorm/
  hnorm/shared_head, `num_nextn_predict_layers=1`) is excluded by the stage
  range machinery, same discipline as #58's dspark blocks. The 24-block ViT
  vision tower (`model.visual.*`) is a separate #57 surface, not part of
  this language-path audit.

## 2. Fleet byte-budget plan (32K context, 4 slots, chunk 2048)

| vector | found | ranges | max stage req | max utilization |
|---|---|---|---|---|
| **3 × 110 GiB** | **yes** | [0,17) [17,31) [31,45) | 108.87 GiB | **0.9898** |
| 3 × 104 GiB | **no** | — | — | honest no-fit |
| 3 × 100 GiB | **no** | — | — | honest no-fit |
| **4 × 84 GiB** | **yes** | [0,13) [13,24) [24,35) [35,45) | 81.41 GiB | 0.9691 |
| 2 × 160 GiB | no | — | — | honest no-fit |

The 3-host shape is essentially at the memory wall (0.9898 utilization on
hosts with ~116–118 GiB available); the 4-host shape carries real margin.
Deployable plan emitted as `host-plan-3x110gib.json` (the ranges JSON the
host command consumes). Cache reservations at 32K×4 slots: 1.3–1.8 GiB per
stage (recurrent linear-attention state + sparse indexer KV mix).

## 3. Real-payload prepare with spot verification (18/18 PASS)

Three single-layer stages covering ALL THREE block kinds, each tensor
compared against an independent `safe_open` read of the same checkpoint:

| stage | kind | checks |
|---|---|---|
| [0,1) + embed | FlashLinearAttention + dense FFN | **5/5** (q/o projections, A_log, dt_bias, embed — fp32-from-bf16 exact) |
| [3,4) | FlashSparseAttention (MLA + indexer) + 288-expert MoE | **9/9** (fp8 kv_a payload byte-exact, E8M0 block scales, kv_b, indexer.wk, experts 0/287 fp8 byte-exact + scales, router gate, e_score_correction_bias) |
| [44,45) + head | last linear-attn MoE layer + lm_head | **4/4** (q_proj, experts.5 fp8 byte-exact, router, lm_head fp32-from-bf16 exact) |

Load times 10–23 s per single layer (fp8→fp32 conversions dominate the MoE
layers); peak RSS 21.2 GiB; cleanup releases back to ~1 GiB. Reproduced
twice (deterministic results both runs).

## Honest scope

- Metadata + payload read + reference residency only: no forward pass, no
  logits, no execution claim. Vision tower and MTP layer unqualified.
- No throughput or quality claim anywhere in this increment.
- All runs CPU-only on labnode3 (the A30/A2 lanes untouched); NAS read-only.

## Next under #57

Teacher-forced single-stage execution parity on real weights (the #705
pattern: prefill/decode parity + reload determinism across both attention
kinds — linear recurrent + sparse indexer), then the multi-host topology
(3- or 4-host plan above; a third/fourth host needs ~110/84 GiB free RAM),
then vision/MTP surfaces.

## Artifacts

`audit-summary.json`, `report-{3x110gib,4x84gib,3x104gib,3x100gib,2x160gib-nofit}.json`,
`host-plan-3x110gib.json`, `cell-a.log`, `glm57_payload_cell_a.py`.
