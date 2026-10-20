# Gate-4 cycle 4: greedy HTTP serving over the real three-rank chain (#536)

Date: 2026-10-20. Checkpoint: Qwen3.5-0.8B
(`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`,
identity fingerprint `ecdd92612ddb79f0df4fc83434943e6a02f3e935c17d0a3a48dda50d739a0312`).

## What ran

`damendaemon chain-serve` (new `chain_server_cli.py`) under torchrun,
`--nnodes=2`, the full three-rank topology — rank 0 coordinator + layers 0-8
on the labnode3 A30, rank 1 layers 8-16 on the A2, rank 2 layers 16-24 +
output head on the desktop RTX 3090 — with the proven `DistributedRuntime`
HTTP scheduler driving `ChainServingPair` → `ChainPipeline` over real NCCL.

**Model identity agreement** (a remaining #536 gate-4 item): every rank
fingerprints its loaded checkpoint (`stage_identity.fingerprint`, computed
on CPU tensors BEFORE weights move to the GPUs), rank 0 broadcasts the
32-byte digest, each rank verifies its local fingerprint against it, and a
MIN ack aborts the whole startup on any disagreement before rank 0 binds
its port. Both hosts printed the same identity and the server started.

## Verified live (HTTP through two hosts, three GPUs)

- `/health` → `{"status": "ok"}`
- `/v1/completions`, raw token-id prompt (seed 16..47), `max_tokens: 12`:
  **`"text": "QRSUVWXYZ012345678"` — 12/12 greedy tokens equal the
  single-rank full-model executor reference** (`gate4-chainserve-reference.json`,
  tokens `87392, 21556, 67848, 15, 16, 17, 18, 19, 20, 21, 22, 23`),
  `finish_reason: "length"`, usage `32/12/44` prompt/completion/total.
  Timing: first token 1.40 s (prefill warmup), 10.7 tok/s decode, batch 1.
- Streaming SSE (`stream: true`) emits the same text as deltas.
- Two SIMULTANEOUS requests: both return the identical 6-token completion
  with `max_batch_size: 2` — concurrent slot batching works across the
  chain; both slots released for reuse (a third request replays
  deterministically; `heartbeats: 35` PING rounds kept idle ranks in
  lockstep over the whole session).
- `/v1/models/status`: `device: "distributed-cuda-chain"`, truthful
  `stage_layers: [[0,8],[8,16],[16,24]]`, `max_batch_size: 2`,
  `packed_prefill: false`, model sha256 = the agreed identity, `completed`,
  `heartbeats` and queue fields live.

Evidence files: `dd536-chainserve-{completion,batch1,batch2,status}.json`,
`gate4-chainserve-reference.json`, rank-0 and rank-2 logs.

## CPU qualification (tests, same code path)

`tests/test_chain_serving.py` (6 tests) drives real HTTP over the
FakeBus chain with the fixture shards: streaming matches the full-model
executor token-for-token with truthful timing/status metadata; concurrent
slots release and reuse; idle heartbeats advance the shared sequence without
touching cursors; invalid requests reject before any chain command; a
worker failure fails the request (500), flips `/health` to 503, poisons
every rank and closes; validation rejects claimed/failed pipelines,
non-contiguous layer bounds and bad identities. Chain modules: 27 tests x4
consecutive OK. Full suite green.

## Implementation notes

- `ChainServingPair.prefill` returns a SCALAR token — the scheduler wraps
  per-item results itself (the two-rank pair contract; returning a list
  poisons the batch with nested-list tokens).
- PING is now a control-only chain round (STOP's sibling): idle heartbeats
  keep all ranks' sequences in lockstep without execution or cursor
  movement. Before this, a PING would have poisoned the epoch.
- Known wart (same as the two-rank `distributed-serve`): a torchrun agent
  SIGTERM kills worker ranks before the coordinator's STOP round
  completes, so forced kills exit with NCCL watchdog noise. The controlled
  path is `--idle-timeout` (drain → STOP → barrier → destroy).

## Scope guards

- Qualified: greedy HTTP completions (stream + batch 2) over the real
  3-rank cross-host chain with byte-equal greedy against the single-rank
  reference, identity agreement, heartbeat liveness, truthful status.
- Not qualified: serving throughput/SLO claims (numbers above are single
  cold-session observations, warmup-inclusive), chunked/interleaved prefill
  (rejected by the runtime), model switching/SWITCH ops, more than one
  alias, any concurrency beyond batch 2.
