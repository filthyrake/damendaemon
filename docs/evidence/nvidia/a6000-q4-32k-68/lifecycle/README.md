# A6000 Q4 32K lifecycle cells — reuse PASSES, cancellation FAILS (#68, 2026-10-31)

The final #68 gate-order items after the C-series (C1 PR #659, C-series PR #660):
reuse and cancellation, one bounded guarded attempt per cell, all pre-declared
on #68 before launch. Runner: `scripts/experiments/hybrid_q4_32k_lifecycle_gate_a6000.py`
— same pinned model/format/context/output/corpus as C1 (slot-0 digest re-verified
against the pinned C1 value at launch), same per-request C1 gates (first output
≤600 s, total ≤1200 s), outer guard 1800 s, near-capacity guard configuration
(reserve 0, thermal opt-in 90 °C) exactly as pre-declared for this card.

## Cell 1 — reuse-identical: PASS

| request | first output | total | tokens | digest |
|---|---|---|---|---|
| 1 (P0) | 154.3 s | 394.1 s | 512 | `3a02b8ce…` |
| 2 (P0 repeat) | 155.3 s | 395.2 s | 512 | `3a02b8ce…` **identical** |

- Repeat reproduces the first completion byte-for-byte at temperature 0
  (compensated Q4 kernels are run-to-run deterministic on this executor).
- Allocator: zero growth across the repeat (21,709,877,248 B both rounds;
  peaks identical at 24,781,027,328 B).
- **No repeat acceleration** (first-output ratio 1.006): the executor
  fully re-prefills identical prompts — no stale prefix state is reused,
  which is the correctness-safe behavior at this qualification level.
- Note: the run's verdict computation hit a harness bug after both
  measurements completed (KeyError: 'request_3' — the shared unpack line
  expected the reuse-suffix round shape); fixed in the same PR. The
  measurements above are from the retained report, unmodified.

## Cell 2 — reuse-suffix: PASS

| request | variant | first output | total | digest |
|---|---|---|---|---|
| 1 (P0) | base | 154.7 s | 394.6 s | `3a02b8ce…` |
| 2 (P_tail) | suffix changed | 155.1 s | 395.0 s | `a9f78585…` **changed** |
| 3 (P_tail repeat) | suffix | 155.1 s | 395.0 s | `a9f78585…` **identical to 2** |

- Changing the final 2,256 tokens (same 30,000-token prefix) changes the
  completion — no stale replay of P0's prefix state.
- The P_tail repeat is byte-identical to request 2.
- All three requests within the C1 gates. Runner verdict:
  `{all_within_gates: True, suffix_changed_output: True, suffix_repeat_identical: True}`.

## Cell 3 — cancel-decode: FAIL (measured finding, plus a server defect)

One 32K/512 request; client disconnected 5.1 s after first streamed
output (first output 154.2 s, disconnect at 159.4 s, 81 chars delivered).

- **The disconnect did NOT cancel the server-side request.** A 512-token
  probe sent immediately after got **503** (both batch slots held: the
  zombie request kept its slot), and the follow-up 32K/64 request also
  503'd. The `/v1/models/status` endpoint went dead as well.
- **Server defect observed (the important finding):** the serve log shows
  an allocation-retry storm after the disconnect — a sequence of growing
  CUDA allocations (228 MB → 770 MB, ~10 s apart, each failing with OOM
  against a card with only ~0.4 GB free) until card exhaustion
  (`gate-cancel-decode.serve.log`). The batch-1 server was effectively
  wedged by a single mid-decode client disconnect; it never recovered
  within the guard window (346 s). Guard phase completed (clean teardown
  of the owned process only).

## Cell 4 — cancel-prefill (batch 2): survivor safe, eventual release, early release FAIL

Two concurrent 32K requests; slot 0 disconnects, slot 1 (survivor) completes.

- Survivor: first output 310.7 s, 512/512 tokens — **unaffected by the
  neighbour's disconnect**, within all C1 gates (matches the C2 numbers).
- Allocator after the round: 26,163,703,808 B = the warmup baseline
  exactly (no leaked KV slot), and the follow-up 32K/64 request passed
  (155.3 s first output, 64/64 tokens) — the slot pool is eventually
  fully usable.
- **Early release FAIL:** the 512-token probe sent at the disconnect
  (~398 s, survivor still decoding) got 503 — the disconnected request's
  slot stayed busy until the zombie completed on its own. No prompt
  disconnect detection.
- Harness caveat (disclosed): urllib's `urlopen` returns only when the
  server sends response headers — on this executor that is at first
  output (~308 s at batch 2), so the "90 s into prefill" disconnect
  actually landed ~90 s into decode. Both cancel cells therefore
  measured mid-decode disconnects; a true mid-prefill disconnect needs a
  raw-socket client (recorded in the runner docstring).

## The combined cancellation finding

Client disconnect is not detected promptly on the Q4 32K executor: the
server-side request runs to completion (zombie slot hold), and at batch 1
the disconnect path triggers a growing-allocation OOM-retry storm that
wedges the whole server. Cancellation is UNSATISFIED at 32K on this
executor — recorded as the measured finding per the pre-declaration (no
sweep, no retry); the disconnect-handling defect is filed for the fix.
Reuse (identical + suffix) is satisfied with correct no-cross-request
semantics.

## Verdict for #68's final item (production-context decision)

32K Q4 on the A6000 executor: C1/C2 within gates, reuse correct, C4
throughput-bound at ~2 slots, cancellation defective. A bounded
production context remains unjustified until disconnect handling is
fixed; recorded on #60/#55.
