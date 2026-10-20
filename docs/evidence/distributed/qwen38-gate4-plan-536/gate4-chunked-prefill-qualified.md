# Gate-4 cycle 5: chunked prefill (PREFILL_CHUNK) over the real three-rank chain (#536)

Date: 2026-10-20. Checkpoint: Qwen3.5-0.8B
(`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`).

## What ran

`damendaemon chain-serve --interleaved-prefill --chunk-size 16` on the full
three-rank topology (A30 coordinator + layers 0-8, A2 layers 8-16, desktop
3090 layers 16-24 + head). `ChainPipeline.prefill_chunk` issues
PREFILL_CHUNK rounds — the coordinator validates the cursor ownership
(start==0 claims a fresh generation; start>0 must equal the slot's committed
cursor), every rank runs its shard with
`positions = arange(start, start+len)`, `prefix_length = start` (the conv /
KV continuation path), and StageState commits
`lengths[slot] = start + rows` only on the MIN ack.

## Verified live (two hosts, three GPUs, chunk size 16)

- 32-token prompt through TWO 16-token chunk rounds, `max_tokens: 12`:
  **`"QRSUVWXYZ012345678"` — byte-equal to the single-rank full-model
  executor reference** (`gate4-chainserve-reference.json`), usage 32/12/44.
  First token 5.47 s (two chunk rounds, warmup-inclusive), 9.3 tok/s decode.
  No throughput claim — cold single-session observation.
- Two SIMULTANEOUS 32-token requests: both chunk-interleaved with decode
  rounds across two slots, both return the identical correct 6-token
  completion with `max_batch_size: 2`.
- `/v1/models/status`: `interleaved_prefill: true`,
  `prefill_chunk_size: 16`, counters live.

Evidence: `gate4-chunked-{completion,batch1,batch2,status}.json`.

## CPU qualification (same code path)

- `test_prefill_chunk_equals_whole_prompt_prefill`: the final chunk's sample
  equals whole-prompt prefill's sample (conv/KV prefix continuation is
  exact), decode continues on the same trajectory from both slots, cursors
  commit on every rank (asserted as final effects — workers commit after
  their MIN-ack contribution), and continuing a mismatched cursor rejects.
- `InterleavedChainServingTests` (whole class re-runs the serving suite with
  `chunk_size=2` + `interleaved_prefill=True`): chunked HTTP completions
  byte-equal the whole-prompt reference, concurrent chunked requests reuse
  slots, status reports the interleaved configuration truthfully.
- Chain modules: 36 tests ×3 consecutive OK. Full suite green.

## Notes

- Intermediate chunk samples are protocol results, never output tokens
  (same convention as the two-rank scheduler path).
- `ChainServingPair.chunk_size` is now a real constructor bound
  (1..4096, fits the session context) instead of the cycle-4 placeholder.
- `ChainDistributedRuntime` accepts `interleaved_prefill` /
  `client_prefill_turns` (the scheduler's chunk-turn machinery); whole-prompt
  prefill remains the default.
- Controlled shutdown is `--idle-timeout` (torchrun SIGTERM races the STOP
  round — known two-rank-parity wart).
