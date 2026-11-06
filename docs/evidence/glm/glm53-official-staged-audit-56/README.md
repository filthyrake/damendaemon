# GLM-5.3 official checkpoint staged-audit (#56, rung 2)

2026-11-06. The official `zai-org/GLM-5.3` release (revision
`ba32f8c19c8738e2c57c71798c601d477a7bf844`) staged to NAS
`/mnt/nas-models/models/GLM-5.3` (user-authorized NAS write): 141 shards +
15 metadata files, 755,663,708,646 bytes on disk (the 755,617,140,416
official payload + ~46.6 MB `hf download` `.cache` metadata dir). Downloaded
via `hf download` on labnode3 at ~200 MB/s.

## Audits (run on labnode3 against the staged checkpoint, no payload reads)

- **names-only** (`names-only.json`): PASS — 117,060 main tensors exact,
  1,569 MTP tensors excluded. Matches the #714 weightless prediction exactly
  (0 missing / 0 extra at the staged-files level too).
- **full header contract** (`header-contract.json`): PASS — every tensor's
  shape + storage dtype matches the pinned contract across all 141 shard
  headers (BF16 floating, F8_E4M3/F8_E4M3FN fp8, F32 scales).

Both audits read only safetensors headers; `weight_values_read: false`,
`model_execution_qualified: false` per the admission discipline.

- `staging-stats.txt` — on-disk size, shard count, download completion.

An identity fingerprint (whole-directory model_id per the #707/#713
convention, `cpu_snapshot_identity.fingerprint`, 760 GiB bound) is running
for the topology rungs; its value lands in the topology cell evidence when
that rung executes.

## Next rungs

Rung 3 payload prepare (stage [0,1) spot-checks; 5.69 GiB layer-0 + embed
per the sweep), rung 4 first execution, rung 5 3-host topology
(labnode3 [0,66) / rocky10 [66,77) / 3090box [77,78) — the #714 plan).
