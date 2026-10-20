# Gate-4 cycle 2c: the ChainPipeline coordinator qualified on the REAL three-rank cross-host chain (#536)

Date: 2026-10-20. Checkpoint: Qwen3.5-0.8B
(`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`).

## What ran

`scripts/experiments/gate4_pipeline_chain.py` under torchrun, `--nnodes=2`,
the full three-rank gate-4 topology:

- rank 0: coordinator + shard 0 (layers 0-8) — labnode3 A30
  (`GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a`)
- rank 1: shard 1 (layers 8-16) — labnode3 A2
  (`GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283`)
- rank 2: shard 2 (layers 16-24) + output head — desktop RTX 3090
  (`GPU-24ef3d4c-51dc-d004-4150-d6a70b05e069`)

NCCL static rendezvous (master 10.30.100.3:29500), `kernels='torch'`, both
hosts torch 2.13.0+cu130 / NCCL 2.29.7. Rank 0 drives unmodified
`ChainPipeline.prefill/decode_batch/close`; ranks 1-2 run unmodified
`ChainStageWorker.run_forever`.

## Result

**`matches_reference: true` — 12/12 greedy tokens equal the single-rank
full-model reference** (`87392, 21556, 67848, 15, 16, 17, 18, 19, 20, 21,
22, 23`), `failed: false`, all three ranks exited cleanly through STOP with
zero tracebacks or watchdog events on both hosts. Full report:
`gate4-pipeline-3rank-report.json`.

- prefill (32 tokens): 1.53 s (warmup-inclusive)
- decode steps 2..12: ~51-59 ms each (first step 511 ms warmup)
- these are wall-clock round trips through the full control plane
  (header broadcast + hidden hops + token broadcast + MIN ack per step)
  over two hosts — NOT serving-throughput numbers.

## Root cause fixed this cycle: cross-node P2P deadlock

The first 3-rank attempt hung: rank 2 (desktop) spun in its cross-node
`return_tokens` P2P send, rank 0 blocked in the matching recv, rank 1 stuck
in the finish MIN-ack (py-spy dump on the live rank-2 process gave the exact
stack). This is the gate-4 cycle 1 finding resurfacing at the pipeline
layer: **per-peer P2P send/recv deadlocks this three-rank ring; the
broadcast-chain pattern is the only transport shape that works here.**

Fix (PR under review): the round choreography became one function all ranks
execute — `run_chain_round` issues the identical collective sequence on
every rank (hidden broadcast src=0, hidden broadcast src=1, token broadcast
src=head, MIN ack), a failing rank emits a fallback payload (zeros hidden /
FAILED_TOKEN tokens) instead of skipping its exchange, and the channel's
P2P send/recv surface is replaced by `exchange_hidden`/`exchange_tokens`.
The CPU FakeBus moved to per-reader FIFO broadcast queues because a round
now has several broadcasts in flight with no reduction barrier between
them. CPU tests: 14 tests x3 consecutive OK, including rewritten
broadcast-exchange tests; the coordinator/worker/middle failure-poison
paths are unchanged in behavior.

## Scope guards

- Qualified: the full 3-rank cross-host pipeline round trip (prefill,
  12 decode steps, STOP close) with greedy equality against the
  single-rank reference.
- Not qualified: HTTP serving lifecycle, concurrency/multi-slot batches,
  any performance claim.
