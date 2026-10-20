# Gate-4 cycle 2b: ChainPipeline coordinator qualified on the real NCCL chain (#536)

Date: 2026-10-20. Checkpoint: Qwen3.5-0.8B
(`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`).

## What ran

`scripts/experiments/gate4_pipeline_chain.py` under torchrun, 2 ranks on
labnode3 (rank 0 = coordinator + shard 0 on the A30
`GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a`, rank 1 = shard 1 + output head on
the A2 `GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283`), NCCL static rendezvous,
`kernels='torch'` (same arithmetic as the single-rank reference).

The coordinator loop under test is the cycle-2 code path end to end:
command header broadcast, MIN ack, hidden P2P, head argmax token return,
cursor commit on every stage, then STOP close. Ranks 1..head run the
unmodified `ChainStageWorker.run_forever`; rank 0 drives unmodified
`ChainPipeline.prefill` / `decode_batch` / `close`.

## Result

`matches_reference: true` — 12/12 greedy tokens equal the single-rank
full-model reference (`87392, 21556, 67848, 15, 16, 17, 18, 19, 20, 21, 22, 23`),
`failed: false`. Full report: `gate4-pipeline-report.json`.

- prefill (32 tokens): 1.72 s (includes lazy Triton/compile warmup)
- decode steps 2..12: ~89–96 ms each (first step 455 ms warmup)
- these are wall-clock round trips through the full control plane
  (broadcast + MIN ack + P2P + token return per step) and are NOT
  serving-throughput numbers.

## Findings

1. **Generation-protocol alignment.** The chain's PREFILL returns the next
   token after the full prompt, while the reference protocol decodes
   `SEED_TOKENS[-1]` at position 32 (the prefill cursor). Using prefill's
   returned token as the first greedy token shifts the whole sequence by
   one position — the first mismatched run was exactly this shift
   (21556, 67848, ... vs 87392, 21556, ...), proving boundary-exact
   arithmetic before the protocol was aligned.
2. **The MIN-ack buffer belongs to the channel**, not the transport: the
   CPU FakeTransport carried `ack` as a test fixture, but
   `NcclChainTransport` had no buffer, so the first real-NCCL round raised
   `AttributeError`. Fixed by allocating the ack tensor in
   `ChainChannel.__init__` on the rank's device.
3. **Runner world-generality.** The qualification runner initially
   hardcoded the three-way layer split and a 3-rank guard; bounds are now
   derived from `world`, so 2-rank and 3-rank geometries share one script.

## Scope guards

- Qualified: 2-rank coordinator loop (A30→A2) with greedy-token equality
  against the single-rank reference.
- Not qualified this cycle: the 3-rank pipeline geometry over NCCL (the
  3-rank transport/control plane was qualified in cycle 1; the desktop
  rank was unreachable this session — password auth path lost, key auth
  rejected — so the 3-rank pipeline run is deferred, not claimed), HTTP
  serving, concurrency, performance claims.
