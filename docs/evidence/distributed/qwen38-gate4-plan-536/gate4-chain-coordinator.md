# Gate-4 cycle 2: chain pipeline coordinator (cross-stage cursor commit)

## What landed

`damendaemon/chain_pipeline.py` wires the cycle-1 control channel
(`distributed_chain.ChainChannel`) to the stage protocol
(`stage_protocol.StageState`/`Command`) and real hybrid shards:

- rank 0 (coordinator) issues encoded commands, executes its own shard
  inline, receives the head's tokens and drives the MIN acknowledgement;
- ranks 1..head run `ChainStageWorker.run_forever` loops: receive hidden
  → `forward_layers` → forward or sample → finish;
- **cursors commit on every stage only after the MIN ack across all
  stages succeeds** — the `hybrid_stage.py` ownership contract ("commit
  request cursors only after every stage succeeds");
- any execution failure poisons the epoch on every rank before any
  cursor advances; workers emit fallback payloads so the whole chain
  reaches the ack instead of deadlocking.

Supported operations this cycle: PREFILL (single slot, packed), DECODE
(batched, divergent slot histories), STOP. The payload contract matches
the two-rank `HostHybridWorker` shape (FP32 `[rows, hidden]` hidden,
argmax token rows returned by the head).

## Coordinator-deadlock fix in the channel

`ChainChannel.finish` previously checked `state.failed` BEFORE the MIN
all-reduce — a worker whose execution failed could not participate, and
its peers blocked in the reduction until the NCCL watchdog killed the
group. `finish` now reduces first, then commits everywhere on success or
poisons every rank on failure. CPU tests cover both outcomes.

## Verification (CPU, deterministic)

`tests/test_chain_pipeline.py` (plus the updated
`tests/test_distributed_chain.py` bus) runs three worker threads against
the coordinator over the in-memory bus with the real `StageState`:

- prefill + 4 decode steps match a full-model single-rank reference
  token-for-token; cursors and sequence numbers advance in lockstep;
- batched decode with DIVERGENT slot histories (3-token vs 2-token
  prompts) advances both cursors correctly;
- injected middle-stage, head-stage and coordinator-stage failures each
  poison all three states, leave every cursor at its pre-round value,
  and surface as a pipeline error;
- STOP closes all worker loops.

The multi-round FakeBus was fixed for reuse (per-reader broadcast epochs,
P2P event reset) — the single-round version silently redelivered stale
payloads on round two.

## What this does NOT claim

- CPU control-plane qualification only: no GPU run, no serving claim, no
  performance claim. The NCCL qualification of this coordinator loop on
  the real three-device chain is the next cycle.
- Full-vocab logit transport (head-side sampler parity with
  `HybridPipeline.prefill`) is a later cycle; the head returns argmax
  ids this cycle.

## Remaining gate-4 work (per issue #536)

- Qualify this coordinator loop on the real labnode3+desktop NCCL chain.
- Batch the per-step control header into the payload transfer (12.96 ms
  broadcast dominates 0.19 ms payload from cycle 1).
- HTTP serving lifecycle over the chain.
