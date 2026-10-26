# Supervised distributed live cell — #608 (live finding: NCCL watchdog beats the in-process supervisor)

Date: 2026-10-25. Host: labnode3 A30+A2 (driver 595.58.03, venv
`/home/damen/dllm-dev/venv`, torch 2.13.0+cu130 / NCCL 2.29.7), repo
staged from main `736070cd` at `/tmp/g608`. User-granted GPU window.

Fixture: Qwen2.5-0.5B (the layer-partition-qualified geometry; the
Qwen3.5-0.8B attempt was refused at validation with its policy reason —
that refusal is itself recorded in the log), ctx 512, batch 1,
unquantized, `distributed-serve --supervise`, coordinator on the A30 and
supervisor-spawned rank-1 worker on the A2 (fresh rendezvous store port
per generation).

## What verified live

- **Supervised startup on the real pair**: coordinator + worker child
  constructed the pair across both GPUs, the all_gather
  transport-version agreement passed (same venv both ranks), and the
  supervised HTTP service reported `200 {'status': 'ok', 'generation': 1}`
  on /health.
- **The supervised entrypoint wiring** (`python -m damendaemon
  distributed-serve … --supervise`) works on real hardware; the
  supervisor's builder spawned the rank-1 child with the matching
  torchrun environment exactly as designed.

## The live finding (negative, decisive)

Failure injection: the rank-1 worker child was SIGKILLed by explicit PID
(verified against /proc cmdline). Observed sequence in
`serve-supervised.log`:

1. `22:01:03` — `damendaemon distributed listening on
   http://127.0.0.1:8400 (supervised)`, health OK, generation 1.
2. `22:01:40` — the coordinator's next heartbeat ALLREDUCE
   (`stage_transport.py:37` ← `distributed.py:284` heartbeat) hit the
   30 s process-group timeout; the **NCCL watchdog dumped debug info and
   took the ENTIRE coordinator process down** ("To avoid data
   inconsistency, we are taking the entire process down").
3. The supervised failed-state path never engaged: HTTP died with the
   process — no honest failed `/health`, no `POST /v1/distributed/rejoin`,
   no respawn. The operator-facing contract ("HTTP stays up in
   failed/rejoining states") did not hold on real NCCL.

**Why the 14 CPU-bus failure-injection tests passed anyway**: the mock
transport raises a catchable per-call exception at the frame deadline; real
NCCL on a blocking collective has an asynchronous process-fatal watchdog
that fires at the process-group timeout (here 30 s, the same deadline the
heartbeat would need to exceed to classify first). On a blocking
heartbeat collective the coordinator learns the peer is dead only when
the collective fails — the same moment the watchdog aborts the process.
The in-process supervisor cannot win that race by deadline tuning alone:
it would need either (a) a non-blocking/`try_wait`-polled heartbeat with
explicit process-group teardown on classification, or (b) external
process-level supervision restarting the coordinator itself.

## Scope after this cell

- Verified live: supervised startup, real-pair construction, the
  all_gather transport-version agreement.
- Still unverified live (blocked by the finding): dead-peer survival with
  HTTP up, bounded operator rejoin, the real respawn path, resume with
  empty slots, generation-bound terminal refusal.
- The rejoin API's 202 path exists and is CPU-bus-tested; it was never
  reachable live because the coordinator process died at the watchdog.

No throughput or availability claims. The window's other cells (#609
MIG, #604 gate 4) are recorded separately.
