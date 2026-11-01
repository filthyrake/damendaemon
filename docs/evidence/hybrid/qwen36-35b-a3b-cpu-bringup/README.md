# Qwen3.6-35B-A3B (qwen3_5_moe) CPU bring-up — #668 cell 1 (2026-11-01)

First real-checkpoint ingestion of the hybrid MoE target through the
damendaemon stack. Pinned snapshot: `Qwen/Qwen3.6-35B-A3B` @ HF revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` (26 BF16 safetensors shards +
config/tokenizer, 64.6 GiB resident FP16), downloaded to
`rocky10-desktop:/home/damen/models/Qwen_Qwen3.6-35B-A3B`. The GGUF form
(`qwen35moe` arch, Q4_K_M) and MTP heads on the same host are recorded but
not used by this cell — the safetensors path carries the transcribed
contract (#14 fixture) and the qualified dense hybrid semantics.

## What ran (CPU only, no GPU touched — `qwen36_35b_cpu_bringup_668.py`)

1. **Contract audit over every shard header** (`hybrid_moe_checkpoint.audit_snapshot`):
   693 main-text tensors exactly matching the transcribed fixture (names,
   shapes, BF16/F32 dtypes); 19 MTP and 333 visual tensors explicitly
   excluded, not validated.
2. **Full load through `ModelWeights`** — the new qwen3_5_moe config
   admission, the upfront contract audit, and the hybrid loader with the
   MoE FFN surface (router, fused 3-D expert stacks, shared expert, combine
   gate). 13.0 s wall, 64.6 GiB FP16; layer schedule `LLLF` × 10 (40
   layers, full-attention interval 4, matching the GGUF metadata
   `full_attention_interval`).
3. **Real-weights routing + FFN smoke through the pinned host reference**
   (`HybridMoeFeedForward`, semantics pinned to the eager Transformers
   modeling file): finite FP32 softmax over the real layer-1 router, distinct
   top-8 selections, renormalized mixture weights summing to exactly 1.0,
   non-degenerate expert usage (top experts 94/6/13 across 9 probe tokens);
   finite layer-1 expert+shared FFN output (RMS 5.0e-3).

The executor-path FFN parity against the same reference is proven exactly on
synthetic weights in `tests/test_hybrid_moe.py` (8 CPU property tests incl.
chunked == whole prefill and decode continuation).

## Honest limits

- **No full-model forward in this cell**: the 64.6 GiB FP16 checkpoint
  exceeds the 44.5 GiB A6000, and quantized qwen3.5-MoE loads are
  deliberately refused until cell 2 implements explicit packed projections
  for expert slices. Full-model numerical gates, context and concurrency
  cells run after that.
- Ingestion/numerical-sanity evidence only — no quality, context,
  concurrency or performance claim.
- Vision (333 tensors) and MTP (19 tensors) remain excluded exactly as the
  contract transcribes.

Artifacts: `qwen36-35b-cpu-bringup.json` (machine record), `bringup.log`
(raw run), runner `scripts/experiments/qwen36_35b_cpu_bringup_668.py`.
