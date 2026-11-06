# GLM-5.3 official-release audit + fleet fit (#56)

Weightless rung-1 audit of `zai-org/GLM-5.3` (official revision
`ba32f8c19c8738e2c57c71798c601d477a7bf844`): contract validation and fleet
feasibility derived entirely from the official `config.json` +
`model.safetensors.index.json` (both fetched from the release; no payload
reads, weights are NOT staged anywhere in the fleet — the NAS has only
GLM-5.3-Flash). Date: 2026-11-06.

## Official-release validation of the existing contract

The pre-existing `glm_checkpoint.py` contract (transcribed from Transformers
v5.15.0 `glm_moe_dsa` at pre-release revision `aca966e4…`) passes the official
release config **unchanged**, and its predicted inventory is exact against the
official index:

- 117,060 main tensors — names set-equality with the official
  `model.safetensors.index.json` weight_map (0 missing, 0 extra)
- 1,569 MTP tensors excluded (layer 78 MTP block: eh_proj/enorm/hnorm style) —
  the contract's MTP exclusion path handles them exactly
- Byte accounting reconstructs the official checkpoint total exactly:
  745,584,469,056 main storage bytes + 10,032,671,360 MTP bytes =
  755,617,140,416 (`metadata.total_size`, 703.7 GiB)
- Official geometry: 78 layers, hidden 6144, 64 heads, MLA (q_lora 2048 /
  kv_lora 512), 256 routed experts + 1 shared, top-8, 3 dense-replace,
  DSA indexer (32 heads, topk 2048), blockwise dynamic E4M3 FP8

Amendments in this PR: revision pin updated to the official release;
`test_pinned_official_release_byte_accounting` added (pinned official total,
main storage bytes, MTP bytes, revision).

## Fleet fit — FEASIBLE (v3, corrected topology: expand labnode3's pool)

Two earlier sweeps were wrong-scoped and are retained per failed-attempt
discipline: `fleet-fit-sweep.json` (four physical CPU hosts only, "no-fit")
and `fleet-fit-sweep-corrected.json` (modeled a phantom "archimedes-vm" as a
separate host from labnode3 — but **labnode3 IS archimedes VM 109**; the
1007 GB Proxmox pool is labnode3's own ceiling). Live-verified 2026-11-06:

- archimedes: 1007 GiB total; other running guests hold 323 GiB; labnode3
  (VM 109 a30-bench, A30+A2 passed through) currently allocated 118 GiB —
  expandable to **683 GiB without touching any other guest** (~740+ trimming
  one idle guest); needs a VM config change + restart
- Physical stage hosts: rocky10-desktop 124 / 3090box 125 / homer 122 GiB
- Dedicated VRAM (executor rungs): 109 GB (A6000 46 + 3090 24 + A30 24 + A2 15)

`fleet-fit-sweep-v4-actual.json` holds the ACTUAL post-expansion plan. Same
retention arithmetic (cache/workspace zero — weights-alone, most generous):

- Retained-converted weights: **698.15 GiB** (694.4 GiB on disk)
- **Applied 2026-11-06: labnode3 (VM 109) expanded 118 → 600 GiB** (614400
  MiB; guest sees 589 GiB after GPU MMIO reservations; archimedes host left
  with ~129 GiB free). The first attempt at 680 GiB wedged QEMU ~100 GiB
  short of host memory and was hard-killed and resized — retained as an
  operational finding (other guests hold ~323 GiB actual vs much larger
  configured allocations; `qm set --memory` takes MiB, not GiB)
- **3-host plan, all within real budgets**: labnode3 [0,66) 585.99 GiB ·
  rocky10-desktop [66,77) 101.29 GiB · homer [77,78) 14.51 GiB (final
  norm/head ride the last stage)

## Consequence for the #56 ladder

The CPU reference pipeline is **not fleet-RAM-gated** — labnode3's pool is
expanded (done, 2026-11-06) and the official 703.7 GiB checkpoint is being
staged to the NAS (`models/GLM-5.3`, user-authorized NAS write). Remaining
rungs: names-only + header audits against the staged checkpoint, payload
prepare, first execution, 3-host topology — the #709→#712 GLM-Flash ladder
recipe. VRAM inventory (109 GB dedicated + 112 GB homer GTT, ~9 GiB/layer
fp8) is recorded for the executor rungs.
