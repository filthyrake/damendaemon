# Gate-4 cycle 1b: real hybrid shards across the three-rank chain (#536)

## What was qualified

`scripts/experiments/gate4_hybrid_chain.py` executes the real Qwen3.5-0.8B
hybrid geometry as three `HybridLayerStage` shards over the qualified
three-rank NCCL chain (cycle 1, PR #556):

- rank 0 (labnode3 A30): embedding + layers 0-7
- rank 1 (labnode3 A2): layers 8-15
- rank 2 (desktop 3090): layers 16-23 + final norm/head + greedy sampling

`reference` mode runs the full model single-rank (eager torch kernels,
packed prefill + 12 greedy decode steps). `chain` mode runs the shards
across the chain and gates on greedy-token equality.

**Result: 12/12 greedy tokens match the single-rank reference**
(`matches_reference: true`, checkpoint sha256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`).

- `gate4-hybrid-reference.json` — single-rank A30 trace
- `gate4-hybrid-chain-report.json` — three-rank chain trace, equality gate

## Findings (all hit and fixed during this cycle)

1. **The reference must pin the same kernel path as the stages.** The
   fused Triton kernels change FP16 reduction order vs eager torch and
   flip near-tie argmax; both sides run `kernels='torch'`.
2. **The residual stream must cross boundaries in FP32.** The executor's
   residual dtype is FP32; rounding to FP16 at each boundary changed the
   logits enough to flip the trajectory.
3. **Every rank must participate in every boundary broadcast** — a
   rank-conditional collective is illegal in NCCL and aborts the group
   (`ncclInvalidUsage` / remote peer closed).
4. **Collectives need explicit `work.wait()`.** NCCL collectives run on
   an internal stream; consumers on the current stream race the payload
   copy without `wait()`.
5. **The head stage must run its layers AFTER receiving the boundary-B
   payload.** Running them before the broadcast (then overwriting with
   the received buffer) silently skips layers 16-23 — detected because
   rank2's post-layer hidden absSum exactly equaled its input absSum.
6. **Per-stage boundary hidden checksums are the debugging tool of
   record**: broadcast the same tensor and compare absSum per stage
   against a no-NCCL single-process chain; equality localizes a
   divergence to a single stage.

## What this does NOT claim

- No serving claim, no throughput/latency claim, no Q4 admission.
- CPU-fixture numerics only (FP16 text, context 512); the 27B Q4
  three-device serving path remains gated on later cycles.
- Greedy equality is a numerics gate, not a plausibility gate: the seed
  tokens are arbitrary raw ids and the reference trace degenerates to
  repeated tokens (expected for arbitrary ids on a small model).

## Remaining gate-4 work (per issue #536)

- Batch the per-step control header into the payload transfer (cycle-1
  finding: 12.96 ms broadcast dominates 0.19 ms payload transfer).
- Hybrid pipeline coordinator wiring (cursor commit across stages).
- HTTP serving lifecycle over the chain.
