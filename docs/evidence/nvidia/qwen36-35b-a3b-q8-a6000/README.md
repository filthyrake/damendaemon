# Qwen3.6-35B-A3B (qwen3_5_moe) Q8_0 A6000 numerical gate — #668 cell 2 (2026-11-01)

First full-model hardware gate of the MoE executor: packed Q8_0 expert
projections on the shared A6000 (`GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed`,
idle at launch), against the cell-1 pinned snapshot
`Qwen/Qwen3.6-35B-A3B` @ HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
(cell-1 evidence: `docs/evidence/hybrid/qwen36-35b-a3b-cpu-bringup/`).

## Runner (`scripts/experiments/qwen36_35b_q8_gate_668.py`)

Two sequential phases in one process, launched under
`scripts.experiments.guard_shared_cuda` (`--thermal-limit-c 90`,
`--gpu-reserve-mib 0`, `--timeout 1800`, report
`guard-q8gate-v2.json`, completed exit 0 in 182 s, peak device 49 °C):

- **Phase A (CPU, unguarded GPU-wise, inside the guarded process)** — eager
  Transformers 5.20.1 `Qwen3_5MoeForConditionalGeneration`, BF16, CPU:
  prefill last-token logits + 12 greedy tokens per prompt
  (`add_special_tokens=False`).
- **Phase B (CUDA)** — damendaemon q8_0 hybrid executor (Triton kernels,
  FP16 compute, `use_graph=False`, context 512, max_batch 1). Packed-state
  assertions verified live: every `*.experts_gate_up`/`*.experts_down`
  stack is int8-packed, every `.router` and `.shared_expert_gate` unpacked.
  Weights 34.8 GiB in 115 s, peak allocation 34.9 GiB.

5 plain-text prompts (5–9 tokens each), same ids on both sides.

## Result

| check | result |
|---|---|
| prefill last-token argmax agreement | **5/5** |
| 12-token greedy stream, token-exact vs BF16 CPU reference | **5/5** |
| logit rel-RMS (prefill last position) | worst **0.1139** (one 9-token prompt), others 0.0323–0.0382 |
| weight / peak allocation | 34.8 GiB / 34.9 GiB |
| guard | completed, exit 0, 182 s, thermal peak 49 °C |

The 64.6 GiB BF16 reference and the 34.8 GiB Q8 candidate cannot coexist on
the 44.5 GiB card (or in one CPU residency), hence the sequential phases.

## Harness defect found and fixed during the gate

The first guarded run reported `greedy False` on all 5 prompts. Root cause
was in the runner, not the executor: phase A built the reference stream as
`[argmax] + generate(new_tokens)`, but `generate`'s first new token IS the
last-position argmax — the reference duplicated its first token and every
elementwise comparison shifted by one. Realignment (`ours[i] == ref[i+1]`)
showed token-exact agreement on all 5 prompts; the runner was fixed (greedy
stream = exactly the generate output, `max_new_tokens=GEN_TOKENS`) and the
gate re-run clean from a regenerated reference. Both the initial artifact
set and this correction are retained per the honest-measurement discipline;
the logit rel-RMS numbers are unaffected (logits comparison was always
aligned).

## Honest limits

- Numerical accuracy/drift gate per the #23 convention — no byte-exactness
  claim across quantized boundaries, and no performance claim of any kind
  (this is not a benchmark).
- 5 short prompts × 12 tokens — prefill logits + greedy continuation only;
  context, concurrency (#59/#60), HTTP lifecycle (#55) and MTP cells come
  later in #668.
- **Q4_0 on the A6000 is not run**: Q4 remains identity-gated to the
  A30/ROCm chain per `hybrid_q4_policy` (the A6000 is not admitted for
  dense Q4 either). The packed Q4 path is covered by CPU property tests
  (`tests/test_hybrid_moe.py`) only — the hardware cell would have to run
  on the ROCm chain.
- Reference is eager Transformers 5.20.1 (installed venv; same family as the
  5.20.2-pinned routing reference in `hybrid_moe_ffn.py`).

## Artifacts (sha256)

- `q8-gate-v2.json` — machine record:
  `c968e02734a67a0bb77776a2f261edabaa7188c735f7f7f09d273ecee7bfd3ef`
- `guard-q8gate-v2.json` — guard report:
  `53a8cf70602c01ce76680b50cef5a7c697532dcaf77d7dac91881928eec10f2a`
- `q8gate-v2.log` — raw run log (guard wrapper + both phases)
- BF16 reference tensors (kept on `rocky10-desktop:/tmp/g668/reference-v2.pt`,
  4.9 MB, regenerable from the pinned snapshot + runner):
  `450cec478777c04296de34d1a0d663bf8a65456d968facb9837030e8fa6b6850`
