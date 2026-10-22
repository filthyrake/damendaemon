# Qwen3.5-MoE checkpoint contract — first #14 real-Qwen-target cell

Date: 2026-10-22 (UTC). The first MoE-foundation cell for the Qwen target named
by #14: a metadata-only checkpoint contract for the `qwen3_5_moe` /
`qwen3_5_moe_text` architecture, transcribed from and pinned against the real
**Qwen3.6-35B-A3B** checkpoint (local NAS snapshot, 26 BF16 safetensors shards,
1,045 index tensors).

## What was validated (metadata only — no payloads, no execution)

- **`damendaemon/hybrid_moe_checkpoint.py`** validates the multimodal root and
  text geometry (hybrid linear/full schedule reconstructed from
  `full_attention_interval`, interleaved mrope section covering the partial
  rotary width, FP32 recurrent-state dtype, untied head, bounds on every
  integer field) and transcribes the 693-tensor main-text inventory:
  - Stacked routed experts per layer: `mlp.experts.gate_up_proj`
    [256, 1024, 2048] (gate+up fused) and `mlp.experts.down_proj`
    [256, 2048, 512], beside the router `mlp.gate.weight` [256, 2048], a dense
    shared expert (512-wide) and a `[1, 2048]` `shared_expert_gate` combine
    gate.
  - Linear-attention layers match the qualified dense Qwen3.5 text contract
    (in_proj_qkv 8192 = 2·16·128+32·128, in_proj_z 4096, a/b 32 rows,
    conv1d [8192, 1, 4], FP32-state dt/A_log [32], norm [128]).
  - Full-attention layers carry the attn-output gate packed in `q_proj`
    ([8192, 2048] = 2·heads·head_dim — the same 2×q convention the dense
    contract encodes; caught by the real-header audit on first pass).
- **Real-checkpoint audits** (labnode3, CPU, header reads only):
  names-only inventory audit — the 693 spec names match the checkpoint's
  non-visual non-MTP tensor set exactly (no missing, no extras); full header
  audit (`qwen3-6-35b-a3b-header-audit.json`) — every shape and dtype
  validated: `main_tensors: 693`, `excluded_mtp_count: 19`,
  `excluded_vision_count: 333`, `weight_values_read: false`,
  `model_execution_qualified: false`.
- **Pinning** (`tests/fixtures/qwen3_5_moe_contract.json`): config sha256
  `93a4693f…efe99` and index sha256 (the only files the contract reads), 26
  shards, and the 693-name fingerprint `dc12cfe8…fda19d0` asserted by
  `tests/test_hybrid_moe_checkpoint.py` (6 tests, stdlib-only, added to the
  CI contract matrix in both plain and `-O` runs). Regenerate the fixture with
  `scripts/experiments/qwen35_moe_contract_fixture.py` (verified
  byte-identical reproduction against the committed fixture).

## Explicitly not claimed

- No weight payload was read; no routing semantics (sigmoid/top-k
  normalization), expert execution, model loading, quality or performance is
  qualified by this cell. MTP (19 tensors) and visual (333 tensors) are
  excluded, not validated.
- The 693-name exact match proves the inventory transcription; it does not
  prove the checkpoint loads or runs.

## Why this checkpoint / next cells

Qwen3.6-35B-A3B is the only modern MoE target with a complete HF-format
checkpoint locally available (the pinned GLM-5.3/GLM-Flash FP8 and DeepSeek V4
Flash checkpoints are not on the NAS; the local GLM dirs are llama.cpp quant
experiments). The hybrid attention machinery is already qualified at 0.8B/2B
dense scale, so the MoE delta is the stacked-expert FFN block above. Ordered
next cells under #14: (1) text-path MoE FFN execution on the packed foundation
with an independent reference at small scale, (2) real-checkpoint expert
kernels (the 67-GiB BF16 weights require staging/quantization decisions before
any single-GPU cell), (3) placement/parallelism per the epic.
