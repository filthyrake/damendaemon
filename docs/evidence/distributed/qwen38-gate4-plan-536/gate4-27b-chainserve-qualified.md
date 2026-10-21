# Gate 4 — 27B Q4 chain HTTP serving qualification (#536 cycle 8)

Date: 2026-10-20 (UTC)
Run: `damendaemon chain-serve` under torchrun on the real three-rank chain
(labnode3 `--nnodes=2 --node_rank=0 --nproc_per_node=2`, desktop
`--node_rank=1 --nproc_per_node=1`, `--rdzv_backend=static
--master_addr=10.30.100.3 --master_port=29500`):

```
chain-serve /tmp/q3827b --quantization q4_0 --context 2048 \
  --rank-budgets-mib 22528,13312,22528 --batch-size 1 --port 8017
```

## What this cycle delivers

- **Production chain-Q4 admission** (`hybrid_q4_policy.chain_anchor`): the
  explicit `DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN=1` opt-in admits the
  qualified three-device shape — a non-A30 tail rank (the A30-primary pair
  anchor cannot hold on the single-GPU desktop) and the chain context cell
  (up to `CHAIN_CONTEXT_LIMIT = 2048`). Every other guard (explicit Q4
  opt-in, accelerator, `qwen3_5_text` architecture, compensated precision,
  no vision) still applies; single-device production admission is
  unchanged. `hybrid_precision.select` carries the same chain context
  bound for compensated Q4.
- **chain-serve quantized path**: `--rank-budgets-mib` (one entry per rank,
  identical on every rank) drives the `choose_stage_splits` planner for the
  layer bounds `[0, 24, 44, 64]` and Triton stages; each rank's admission
  env declares only its **node-local** budget slice (empty_cache walks one
  CUDA ordinal per entry). Context is admitted to 2048 for `q4_0` (512 for
  unquantized, unchanged); `q8_0` chain serving is rejected (unqualified);
  quantized serving keeps a single resident model (no SWITCH table).

## Result (batch 1, context 2048 — the qualified cell)

- `/health` ok; two consecutive raw-token-id completions (seed 16..47,
  `max_tokens 12`) both **byte-equal** the in-process A30+A2 reference of
  the same quantized path under the **standard autoregressive protocol**
  (`gate4-27b-serve-protocol-reference.json`): greedy
  `[70182, 25223, 87197, 62, 63, 62903, 59656, 59726, 3994, 1, 198, 220]`
  → text `QRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxy"\n ` — deterministic
  slot release and reuse across requests.
- `/v1/models/status` truthful: `distributed-cuda-chain`, context 2048,
  `q4_0`, stage layers `[[0,24],[24,44],[44,64]]`, agreed identity
  sha256, `model_switches: 0`.
- Per-rank device memory at serve: A30 ~8.3 GiB, A2 ~4.6 GiB, 3090 tail
  ~5.5 GiB staging peak. No throughput/SLO claim.

## Protocol-convention finding (recorded honestly)

The first comparison against the cycle-7 runner reference "diverged" at
token 6. Root cause — a **decode-input convention difference**, not numeric
divergence (A/B probe `gate4-27b-protocol-abprobe.json`, one chain, both
protocols):

- The chain/runner scripts (cycles 2b–7) decode with **`prompt[-1]` re-fed
  at the prefill cursor** — the input stream becomes `[prompt, prompt[-1],
  ...]`. The 0.8B ladder's byte-equality claims used this convention on
  both sides (runner and reference), so they are internally consistent,
  but the stream doubles the last prompt token.
- `ChainCompletion`/`HostCompletion` (the qualified serving contracts
  since #511) decode with the **generated token** — the standard
  autoregressive stream `[prompt, t0, t1, ...]`.
- The 0.8B's rigid pattern continuation made both conventions produce the
  SAME greedy sequence, masking the difference; the 27B exposes it. On the
  same 3-rank chain: runner protocol → `[70182, ..., 271]` (matches the
  cycle-7 reference), serving protocol → `[70182, ..., 220]` (matches the
  standard-protocol in-process reference).

The serving path is correct as-is (it mirrors the pre-chain qualified
`HostCompletion` contract); future chain qualification runners should use
the standard protocol (`decode(generated)`) so runner evidence and serving
evidence agree. Cycle-7's claim stands as written (byte-equal under the
runner convention on both sides) but should be read as transport
qualification, not serving-protocol qualification.

## Admission substitutions: none

Unlike cycle 7's probe, this cycle required **no monkey-patching**: the
chain opt-in is real production policy, and the served run used it
directly. The runner scripts' probe admission remains in
`scripts/experiments/` for reproduction of the cycle-7 evidence.

## Reproduction

- Reference: `PYTHONPATH=<repo> python scripts/experiments/g4_27b_serve_reference.py`
  logic (in-process A30+A2 pair, chain opt-in, standard protocol) →
  `gate4-27b-serve-protocol-reference.json`.
- A/B probe: `gate4-27b-protocol-abprobe.json` (one 3-rank chain, both
  protocols; runner-protocol branch matches the cycle-7 reference,
  serving-protocol branch matches the standard reference).
- Serving: the launch line above; completions `gate4-27b-chainserve-
  completion{1,2}.json`, status `gate4-27b-chainserve-status.json`.
- Tests: `tests/test_hybrid_pipeline_quant.py` (chain-anchor admission:
  tail rank admitted, rejected without the opt-in, context bound 2048,
  standard guards retained, compensated select at 2048) and
  `tests/test_chain_server_cli.py` (CLI validation + node-local budget
  slicing); chain quartet + policy suites green; full suite OK
  (skipped=5).
