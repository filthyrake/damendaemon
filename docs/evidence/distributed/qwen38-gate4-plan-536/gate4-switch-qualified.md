# Gate-4 cycle 6: resident model switching (SWITCH) over the real three-rank chain (#536)

Date: 2026-10-20. Base checkpoint: Qwen3.5-0.8B
(`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`).
Alt model: the same snapshot with a deterministic +0.05 nudge on
`model.language_model.layers.0.mlp.gate_proj.weight` (sha256
`83b47ed4c1f76b5af236b1df…`, byte-identical copies on both hosts).

## What ran

`damendaemon chain-serve --model alt=/tmp/dd536-altmodel` on the full
three-rank topology (A30 coordinator + layers 0-8, A2 layers 8-16, desktop
3090 layers 16-24 + head). Every rank loads both models, fingerprints both
on CPU before any weights move, and the per-alias identity digests are
broadcast + MIN-acked before the port binds. `StageState(models=2)` admits
the SWITCH command; the shard table keeps each rank's layer-range weight
dict per model on CPU (index 0 = the resident model, so switching back
restores exact weights).

The SWITCH round: the coordinator validates (known index, not current, all
slots released), issues the encoded command; EVERY rank copies its shard
in place (geometry validated before any copy), then the MIN ack commits
`model_index` + a fresh epoch (generations reset) everywhere or poisons
on any failure. `ChainServingPair.switch_model` drives the scheduler's
deferred-switch path, and `/v1/models/status` reports per-alias residency.

## Verified live (two hosts, three GPUs)

Three requests in sequence — `model`, `alt`, `model`:

- base → `"QRSUVWXYZ012345678"` — the recorded single-rank reference, exact
- alt → `"QRS\n123456789"` — **byte-equal the ALT's own single-rank
  executor reference** (tokens `48, 11122, 198, 16…24`; every rank's shard
  really swapped)
- back to base → `"QRSUVWXYZ012345678"` — the original weights replay exactly

Status after: `model_switches: 2`, `current_model: model`,
`[('model','gpu'), ('alt','host')]`, `completed: 3`, `heartbeats: 43`,
`failed: False`. Evidence: `gate4-switch-{base,alt,back,status}.json`,
`gate4-switch-alt-reference.json`, rank-0 log.

## CPU qualification (same code path)

- `test_switch_model_swaps_resident_weights_and_epoch`: switching with a
  live slot rejects; switching to the current index is a no-op; after the
  switch the epoch resets and every rank's model_index commits (asserted
  as final effects); the alt model's first sample equals its reference
  executor; switching back replays the original sample.
- Harness fix found by the live prep: on CPU the stage ALIASES the model
  weights, so shard tables must hold CLONES (a switch overwrites the
  source tensors in place; production GPU stages own separate device
  tensors and cannot alias).
- `SwitchingChainServingTests`: the full HTTP serving suite over a
  two-model chain — alias-routed requests switch through the scheduler's
  deferred path, outputs match each model's reference, status reports
  per-alias residency/identities/switch count, unknown aliases reject
  before any command.
- 45 chain-module tests ×3 consecutive OK.

## Operational incident (recorded honestly)

The alt-model prep script's first version wrote through the SOURCE path —
the HF snapshot's safetensors is a symlink into the blob store — and
corrupted the pinned 0.8B blob on labnode3 (perturbed twice). Recovery:
verified the desktop's blob was pristine (sha256 == blob name), scp'd it
back, verified the labnode3 base hash restored to `04b1c301…`. The script
now operates only on the target copy. No evidence run before the incident
is affected (all earlier references were recorded pre-corruption, and the
restored base reproduces them exactly — the `back` request above).

## Scope guards

- Qualified: resident switching between two identical-config models with
  byte-equal outputs vs each model's single-rank reference, identity
  agreement per alias, deferred alias routing under the HTTP scheduler.
- Unquantized only this cycle: packed Q8/Q4 scales bind to weight tensor
  ids inside the quantized Linear (the two-rank stage_rebind machinery
  owns that path). One additional alias (`--model` accepts one entry),
  matching the two-rank CLI bound.
