# Gate 4 — 27B Q4 three-device chain qualification (#536)

Date: 2026-10-20 (UTC)
Run: `scripts/experiments/gate4_27b_chain.py` under torchrun
(`--nnodes=2 --node_rank=0 --nproc_per_node=2` on labnode3,
`--node_rank=1 --nproc_per_node=1` on the desktop,
`--rdzv_backend=static --master_addr=10.30.100.3 --master_port=29500`).

## Topology

| Rank | Host / device | Layers | Budget |
|------|---------------|--------|--------|
| 0 (coordinator) | labnode3 A30 | 0–24 | 22528 MiB |
| 1 | labnode3 A2 | 24–44 | 13312 MiB |
| 2 (+ output head) | desktop RTX 3090 | 44–64 | 22528 MiB |

Checkpoint: Qwen3.8-27B (`1d4bf0f2`, 30 files, 52 GiB; staged to labnode3
`/tmp/q3827b` with all 18 `model-*.safetensors` sha256-verified byte-identical
to the desktop copy). Quantization `q4_0`, compensated precision, Triton
stages, context 2048, max_batch 1.

Layer bounds come from the same `choose_stage_splits` storage planner as the
gate-2 pair, evaluated with the three per-rank budgets — deterministic, so
every rank derives the identical split `[0, 24, 44, 64]` (64 layers).

## Result

12/12 greedy tokens byte-equal to the reference
(`gate4-27b-reference-report.json`): the A30+A2 two-stage in-process
pipeline (gate-2 convention) of the SAME quantized path — seed
`range(16, 48)` prefilled whole, then 12 greedy decodes of `SEED[-1]`.

- chain greedy = reference greedy = `[70182, 25223, 87197, 62, 63, 65234,
  44576, 92, 93, 198, 71093, 271]` → `matches_reference: true`,
  `failed: false`
- prefill (32 tokens): 18.7 s; decode: first step 4014 ms (Triton compile),
  then 139–145 ms/step steady
- rank-0 peak device allocation: 7.42 GiB (budget 22528 MiB)
- all three ranks exited cleanly; both GPUs idle after teardown

Timings are warmup-inclusive (first decode includes Triton compilation) and
are **not** serving throughput claims. Steady decode (~140 ms/step) is
comparable to the two-device in-process reference (~160 ms/step steady).

## Admission substitutions (probe convention, recorded honestly)

Production admission in `hybrid_q4_policy` is unchanged. The runner's
admission patch follows the `hybrid_q4_context_probe` convention extended
one guard, both recorded as part of what this gate qualifies:

1. **Context admission** — the production 512 bound is substituted with the
   probe's 512..32768 window (same as gate 2). Opt-in, accelerator,
   architecture (`qwen3_5_text`), compensated-precision and vision guards
   are replicated intact.
2. **A30-primary pair anchor** — the anchor requires `cuda:0` to be an A30;
   the desktop's RTX 3090 tail rank cannot satisfy it, so the chain-probe
   admission admits that rank explicitly. The three-device chain is the
   first Q4 workload on the 3090; this gate (byte-exact transport of its
   stage outputs into the agreed greedy sequence) is that device's
   qualification evidence.

`StageState` uses the explicit `context_limit` metadata admission (up to
2**20 per `stage_protocol`); the merged `chain-serve` CLI remains capped at
context 512 — 27B chain HTTP serving at 2048 context is future work, not
claimed here.

## Reproduction

- Reference: `g4_27b_reference.py` logic (in-process A30+A2 pair) →
  `/tmp/dd536-27b-reference.json` on labnode3.
- Chain: `torchrun ... /tmp/g4_27b_chain.py` on both hosts (runner source
  committed at `scripts/experiments/gate4_27b_chain.py`) →
  `/tmp/dd536-27b-chain.json` on the rank-0 host.
- Evidence: `gate4-27b-chain-report.json`, `gate4-27b-reference-report.json`.
