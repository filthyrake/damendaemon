# #58 — Two-process host topology on real official weights: first full-model generation, "Hello!"

**Fifth physical-qualification increment** (after #702/#703/#704/#705): the
#703 2×96 GiB fleet plan executed as a REAL two-host topology — both stages
of the official `DeepSeek-V4-Flash-0731` resident simultaneously (148 GiB
across hosts), TLS-authenticated, whole-snapshot source-verified on each
side, native chat turn completed end-to-end.

## The run

| | coordinator (labnode3) | worker (rocky10-desktop) |
|---|---|---|
| stage | [0,22) — embedding boundary | [22,43) — head boundary |
| source | NAS original (read-only CIFS) | byte-identical LAN copy (166,898,669,876 B, 150 files, ~12 min at ~230 MB/s) |
| residency | 76.1 GiB (118 available) | 73.0 GiB (116 available) |
| identity | model_id `7727d9c1…4b83c`, verified | same model_id, verified |

- Transport: pinned-cert TLS 1.3 over the LAN (10.30.100.3 ↔ 10.40.100.30),
  ephemeral 2-day throwaway CA (conventions from `tests/tls_credentials.py`).
- Both sides hashed the WHOLE snapshot (166.9 GB) before loading
  (`--verify-model-id --identity-max-mib 160000 --identity-timeout 2400`);
  both `model_identity_verified: true`; matching topology identity
  `69930596…9b66`.
- Chat turn (native DeepSeek chat, real tokenizer + template):
  prompt `"Hello."` → 6 prompt tokens → **2 generated tokens: "Hello!"** →
  `finish_reason: "length"`, `usage: {prompt_tokens: 6, completion_tokens: 2}`.
  ~50 min wall for the turn (tile-wise fp32 CPU reference, 16/32 threads) —
  the reference executor's cost, not a serving claim.
- Clean teardown: worker `{"event": "stopped"}`, both hosts returned to full
  availability; NAS never written; A6000/A30 untouched (CPU-only throughout).

## Fixture-scale walls found and dissolved (this PR)

The topology command could not run real weights — five separate fixture-scale
bounds refused at different layers, each dissolved structurally (no policy
change, defaults unchanged):

1. `HostTensorChannel` op deadline ≤ 300 s → ≤ 86400 s (real chunks compute
   for minutes between transport ops; dead peers still fail on EOF).
2. CLI `--timeout` ≤ 30 s → ≤ 86400 s (same).
3. CLI `--weight-mib`/`--cache-mib` ≤ 2048 → ≤ 2²⁰ MiB (matching the #703
   planner amendment; workspace/output caps unchanged).
4. `host_tls` accept/handshake wait ≤ 30 s → ≤ 86400 s (stages load for tens
   of minutes before a peer connects).
5. `build_reference_host_worker` byte bounds ≤ 2³¹ → ≤ 2⁴⁰; plus the topology
   command now FORWARDS the CLI identity bounds to the builder (it silently
   used 2 GiB/60 s defaults, refusing the very snapshot the coordinator had
   verified with the same flags).

One honest runtime refusal retained as a finding: `--weight-mib 78000` was
refused by the loader's budget admission (stage [0,22) needs 75.61+0.99 =
82.3 GiB retained+overlap); relaunched at 84000.

Pinned by `test_real_scale_stage_budgets_and_transport_deadlines_pass_preflight`
(real-scale budgets + day-scale deadlines + identity forwarding through to
the builder) and the updated `2**20+1` weight boundary case. 56 targeted
tests pass.

## Honest scope

- 2 generated tokens through 43 real layers is an execution/transport/
  identity proof, not a quality verdict — `official_model_qualified: false`
  in every event by design. Practical context (32K), quality tiers, serving
  SLOs and the dspark/MTP surface remain open under #58.
- The decode-vs-prefill numerical parity of this pipeline was pinned at
  2–3e-7 in the #705 single-stage cell; this cell adds the distributed
  topology, transport, and identity layers on top.

## Artifacts

`coordinator.log` / `worker.log` (full event streams), `coordinator.sh` /
`worker.sh` (exact invocations), `plan.json` (the #703 2×96 ranges),
`peers.json`.
