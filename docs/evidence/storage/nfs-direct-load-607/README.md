# Direct checkpoint loads from a network share (#607)

Date: 2026-10-23. Host: the development Mac (macOS, arm64), reading a real
SMB share (`//damenknight@nas._smb._tcp.local/llm-models` on
`/Volumes/llm-models`). Read-only access; the share was never modified.

## Change under test

`damendaemon.network_filesystem` classifies mounted filesystems by parsing
the host mount table once per process (`/proc/mounts` on Linux, `/sbin/mount`
on macOS) and stating each network mountpoint to map its `st_dev`. File
identity at all six producer sites (`gguf_model.file_identity`,
`safetensor_header.file_stamp`, `cpu_snapshot_identity._stamp`,
`awq_model.identity`, `reference_pipeline_build.snapshot_inventory`,
`weight_cache.PreparedCache.snapshot`) drops `st_ctime_ns` on those devices
and keeps the strict size/mtime/ctime contract everywhere else. Devices that
cannot be classified fail closed to the strict contract.

## Result: discovery and header reads complete from the share

Probe: `PYTHONPATH=. python3 scratch/nfs_direct_load_probe.py` from the
repository root (script retained in the session scratch, reproduced inline
below). No GPU, no staging copy.

- `Qwen3-4B-Thinking-2507-Q4_K_M.gguf` (2.49 GB host bytes, quantization
  `mixed`): `gguf_model.identity` stable across repeated reads (arity 4,
  network mode); full `gguf_model.plan` — shard discovery plus BPE tokenizer
  construction — completed directly from the share in 3.9–4.6 s with the
  post-plan identity unchanged. Before the change this path aborted with
  `GGUF file changed while discovering shards` on read-side ctime jitter.
- `qwen2.5-0.5b-instruct` snapshot: `checkpoint.inspect_snapshot` (header
  read through `read_header_json` plus tensor inventory) completed in
  0.07–0.18 s with no `checkpoint changed while reading header` abort;
  the single shard parsed normally.

## Local strictness preserved

`tests/test_network_filesystem.py` (16 tests) covers both modes with a local
temp file whose ctime is bumped via `os.utime` with identical atime/mtime:

- strict local: GGUF, safetensors, AWQ, CPU-snapshot and pipeline-inventory
  identities all change when only ctime changes (ctime stays load-bearing);
- network mode: the same touch leaves every identity equal, while size or
  mtime changes still raise;
- mount-table parsing (macOS `mount` output with real-world smbfs/nfs lines,
  `/proc/mounts` name unescaping, selection, subprocess failure) and the
  device map (real mountpoint stat, unmapped-device fail-closed);
- a live test (skipped where absent) asserts the real `/Volumes/llm-models`
  share classifies as network while local paths stay strict.

Focused batch with the touched modules' existing suites:
`tests.test_network_filesystem tests.test_cpu_snapshot_identity
tests.test_weight_cache tests.test_awq_model tests.test_checkpoint_inventory
tests.test_gguf` — 63 tests, OK. Full-suite result recorded in the PR.

## Scope and not claimed

- Metadata identity only: the receipts' documented threat model (ordinary
  concurrent replacement/modification, not hostile writers) is preserved via
  device/inode/size/mtime. A cross-run per-shard content-digest cache remains
  deliberately unimplemented; that decision and rationale live on the issue.
- One host and one share class (SMB). NFS paths are covered by the same
  classification set and unit tests, not by a live NFS mount in this cell.
- No throughput, streaming-bandwidth characterization or prepared-weight
  cache measurements over the share — those are the issue's remaining gates.
