# GLM-5.3-Flash first 3-host topology cell (#57)

First full-model native chat generation on real GLM-5.3-Flash weights across a
three-host CPU reference topology: labnode3 (coordinator, stage 0 [0,17)),
homer (worker, stage 1 [17,31), WiFi IP 10.30.100.216), rocky10-desktop
(worker, stage 2 [31,45), 10.40.100.30). Date: 2026-11-05.

## Result

- All three stages READY with shared topology identity `804d8b80dda3f82a
  2cb894ebd4f9ac5d5b69866710ad2624aa8aba9a67453af0` and `model_identity_
  verified: true` everywhere; source model_id `3160b6d537291fd28c80c4ec
  9f6c603bd57dddbfaefa1f218e4766637f635a88` (whole-directory identity hash
  of the 328,366,173,639-byte official checkpoint, verified independently on
  the NAS original (homer, labnode3) and on rocky10's LAN-copied snapshot
  `/home/damen/glm53f-snapshot` before launch).
- Native chat turn: prompt "Hello." → 14 prompt tokens → 2 completion tokens
  `{"reasoning_content": "Let"}` + `{"reasoning_content": " me"}`,
  `finish_reason: length` (GLM's chat template opens a reasoning segment;
  the budget cap stopped it at 2 tokens, as configured).
- Clean teardown on all three hosts (`{"event": "stopped"}` on both workers,
  coordinator exit). GPUs untouched throughout; NAS never written.

## Cell parameters

plan.json ranges [[0,17],[17,31],[31,45]]; context 128, slots 2, epoch 1,
prefill chunk 8; per-stage budgets weight 112640 MiB / cache 4096 MiB /
workspace 8192 MiB / output 64 MiB / frame 64 MiB; identity bounds
`--verify-model-id --identity-max-mib 330000 --identity-timeout 3600`;
timeout 7200 s; TLS: ephemeral 2-day CA + per-worker server certs with SANs
for each host's DNS + LAN IP (tests/tls_credentials.py conventions).
Weights: homer and labnode3 read the NAS mount directly; rocky10 read the
byte-identical LAN copy (its whole-snapshot model_id matched the original).

## New wall found and dissolved: fixture-scale stage-cache ceiling

First complete launch reached all three hosts' stage construction and
refused with `ValueError: invalid CPU stage slot controls`: the cell's
`--cache-mib 4096` (2**32) crossed the 2**31 `max_cache_bytes` ceiling in
`CpuStageSlots.__init__` (and the mirrored ceilings in `FlashModel.new_state`,
`FlashLinearAttention.new_state`, `FlashSparseAttention.new_state`). The
official 3x110 GiB fleet plan already reserves 1,844,051,968 bytes of slot
cache for stage [0,17) alone (context 32768, slots 4) — within 15% of the old
ceiling — so real-scale cache budgets were structurally refused, the same
fixture-scale cap family as #703/#707/#712. Dissolved in the accompanying
commit: ceiling 2**31 → 2**40, defaults unchanged, pinned by
`test_real_scale_stage_cache_budgets_are_admitted`
(tests/test_glm_flash_stage_slots.py): official-plan aggregate reservation
admitted, aggregate-1 refused, the cell's own 4096 MiB budget admitted,
2**40+1 still refused, and the model-level `new_state` bound pinned the same
way.

## Operational findings

- homer's 10.30.100.216 is its WiFi interface (wlp195s0); firewalld rejected
  the coordinator's connection from labnode3 (same-subnet "No route to host").
  Opened with a runtime-only rich rule scoped to labnode3's source address
  (reverts on reboot, not persisted).
- Never probe a listening worker port with a TCP connect (e.g.
  `echo > /dev/tcp/…/9471`): the worker accepts it as the coordinator
  connection, the probe's close surfaces as an SSL EOF and the worker dies.
  This killed both workers once mid-cell; relaunch sequence that works:
  start workers, wait for both `listening` events, then start the coordinator.
- First-token latency through the full 3-host chain: ~90 min wall for the
  14-token prefill + 2 decode tokens (45 layers, fp32 CPU reference; workers
  peaked ~800% CPU while the coordinator blocked in recv between handoffs —
  the pipeline's layer handoff works as designed).

## Files

- coordinator.log / worker-homer.log / worker-r10.log — full run logs
- plan.json, peers.json — topology plan and peer endpoints
- worker-homer.sh, worker-r10.sh, coordinator.sh — exact launch scripts
  (MODEL_ID placeholder values shown as launched, model_id embedded)
