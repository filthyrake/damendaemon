# #51 two-rank pair gate: A30 + A2 distributed-serve (passed)

## Result

`distributed_pair_gate` completed under the shared-CUDA two-device guard on
labnode3 (2026-10-17):

- Devices: A30 `GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a` (rank 0) +
  A2 `GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283` (rank 1), pinned via
  `CUDA_VISIBLE_DEVICES` by `guard_shared_cuda`.
- Model: Qwen2.5-0.5B-Instruct snapshot `7ae557604adf67be50417f59c2c2f167def9a775`,
  q4_0, context 512, batch 4, torch 2.6.0+cu124.
- Two greedy generations of `What is 17 * 23? Answer with one number.`
  (16 tokens each) returned byte-identical text:
  ` To find the product of 17 and 23, we can simply` — `deterministic: true`.
- Wall clock: 0.238 s (first) and 0.089 s (second) per generation request;
  whole guarded run 50.9 s including model load and health wait.
- `pair-gate-guard.json`: guard phase `completed`, no thermal or reserve
  violations. `pair-gate-serve.log`: full serve log.

## Files

- `pair-gate.json` — gate report (`completed: true`, `deterministic: true`, generations)
- `pair-gate-guard.json` — guard report (phase `completed`, 50.9 s)
- `pair-gate-serve.log` — two-rank serve log (see note below)

## Note on the serve log's NCCL watchdog lines

The log ends with NCCL watchdog ALLREDUCE-timeout errors. These occur during
teardown only: the gate calls `server.terminate()` after success and a rank in
a mid-collective shutdown triggers the watchdog. They are not evidence of a
runtime failure — the gate report recorded both generations complete and
deterministic before termination.

## Bugs found on the way

1. Rendezvous port contention (default 29500) caused silent torchrun hangs —
   harness moved to 29531 (PR #514).
2. `distributed_pair_gate.health()` took an absolute `time.monotonic()`
   deadline but was passed a duration, so every guarded run failed its health
   check instantly — fixed to `health(time.monotonic() + 900)` (PR #515).
   All earlier "hung pair gate" runs trace to this one-line bug, not NCCL or
   the hardware pair.
