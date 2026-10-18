# #489 live two-rank A→B→A resident model switching (passed, bounded scope)

## Result

`distributed_switch_gate` completed under the shared-CUDA two-device guard on
labnode3 (2026-10-17), exercising the live SWITCH stage command from #487 on
real weights across both ranks:

- Devices: A30 `GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a` (rank 0) +
  A2 `GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283` (rank 1), torch 2.6.0+cu124.
- Sequence: generate on alias `model` (A1), alias `alt` (B — live switch
  A→B), alias `model` (A2 — live switch B→A). All three returned 16 finite
  greedy tokens; the two A generations are byte-identical
  (`deterministic: true`) and the B generation answers its distinct prompt.
- `/v1/models/status` residency tracked the requests exactly
  (`model: gpu / alt: host` → `alt: gpu / model: host` → back), with the
  switch counter incrementing once per requested model change (counter
  observed at 2 on the first read — startup/warmup switches — then 3, 4).
- Guard phase `completed` in 65.2 s; whole gate ~65 s including two-rank load.
  Per-request wall times recorded in `switch-gate.json`; shared-host latencies
  are not performance evidence.
- The NCCL watchdog lines at the end of `switch-gate.serve.log` are teardown
  noise from `server.terminate()` after success, as in the #51 pair gate.

## Bounded scope: same-checkpoint aliases

Both aliases point at the same real checkpoint (Qwen2.5-0.5B-Instruct
`7ae557604adf67be50417f59c2c2f167def9a775`). The identity contract requires
identical decoder configurations for resident switching. The only locally
available second checkpoint (Qwen2.5-0.5B base, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`) is correctly rejected by that
gate: its config differs in `eos_token_id` (stop semantics the runtime
actually reads, `qwen2.py`) plus metadata keys. Normalizing those in a staged
copy would have misrepresented the checkpoint's stop semantics, so the
first-pass gate ran with same-checkpoint aliases; the switch machinery
(drain, peer identity agreement, SWITCH, epoch-advancing commit, both-rank
`copy_stage_weights`) is fully exercised live, but the distinct-weights
variant remains open for when a second decoder-identical checkpoint is
staged. A rejected distinct-config attempt is retained in the serve log
history (`ValueError: distributed model switching requires identical decoder
configurations`, 18:54 UTC).

## Files

- `switch-gate.json` — gate report (generations, statuses, determinism)
- `switch-guard.json` — guard report (phase `completed`, 65.2 s)
- `switch-gate.serve.log` — two-rank serve log (teardown watchdog noise only)
