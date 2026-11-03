# #58 — Official DSV4F-0731 host byte-budget plan: 147.94 GiB across the CPU fleet (header-only, no execution)

**Second physical-qualification increment**: after the pinned contract audit
(#702) passed on the real weights, the declared-storage byte-budget planner
now produces real fleet stage plans against the official checkpoint on the
NAS — header-only, read-only, zero payload reads, zero execution claims.

## The structural amendment (same discipline as #702)

The planner was fixture-scale: `plan_reference_stages` capped
`max_weight_bytes`/`max_cache_bytes` at 2 GiB and `frame_bytes` at 64 MiB,
and the CLI capped `--weight-mib`/`--cache-mib` at 2048 MiB. The official
release retains **147.94 GiB (~3.3 GiB/layer)** — every real candidate stage
was structurally refused regardless of the budget vector, and the 32K
chunked-activation frame (2048 rows × 16384 width × 4 B = 128 MiB) exceeded
the frame cap on its own.

Widened to real-scale structural bounds (no policy change):
- library: weight/cache ≤ 2**40 (matches the existing per-host budget cap);
  frame ≤ 2048 MiB; workspace/output caps unchanged (genuinely small allowances)
- CLI: `--weight-mib`/`--cache-mib` ≤ 2**20 MiB (matches `--budget-mib`),
  `--frame-mib` ≤ 2048 MiB; defaults unchanged (fixture-scale runs identical)
- pinned by `test_real_scale_stage_budgets_are_admitted_for_official_release_planning`:
  real-scale options admitted, 2**40+1 / frame+1 still refuse, CLI accepts
  real-scale values; the stale `max_cache_bytes=2**31+1` boundary case updated
  to the new cap.

## Real measurements against the official checkpoint

Inspection (labnode3, `/home/damen/dsv4f-venv`, NAS read-only): retained
**158,844,075,484 bytes = 147.94 GiB** across 43 layers + embedding (1.97
GiB) + head (1.97 GiB); per-layer 3.32–3.37 GiB; largest single-file
conversion overlap 0.99 GiB. Cache reservation at 32K context × 4 slots:
452 MiB/slot, 1,810 MiB total (ratio-4 layers dominate at 20.3 MiB/slot).

Fleet plan sweep at `--context 32768 --slots 4 --prefill-chunk 2048`:

| vector | found | ranges | max stage req | max declared utilization |
|---|---|---|---|---|
| 3 × 64 GiB | yes | [0,14) [14,29) [29,43) | 51.78 GiB | 0.809 |
| **2 × 96 GiB** | **yes** | **[0,22) [22,43)** | **78.26 GiB** | **0.8152** |
| 2 × 80 GiB | yes | [0,22) [22,43) | 78.26 GiB | 0.9783 |
| 2 × 75 GiB | **no** | — | — | honest no-fit boundary |

Fleet context: hosts carry 111–118 GiB free RAM each; 2 × 96 GiB fits two
hosts with reserve, 3 × 64 GiB fits three with wider margin. The 2×96 plan
is emitted as `host-plan-2x96gib.json` — the exact ranges JSON the existing
host command consumes.

## Honest scope

- Declared-storage planning only: `execution_admitted: false`,
  `source_model_identity_verified: false`; overhead, derived preparation
  tensors and unmeasured temporaries explicitly excluded in every report.
- Nothing was deployed; no host process was started; the NAS was never written.
- Env note: the CPU reference pipeline needs Python ≥ 3.10 — a dedicated
  `/home/damen/dsv4f-venv` (python3.11 + CPU torch 2.15 + safetensors +
  tokenizers) now exists on labnode3 for this lane (waldo-venv is 3.9).

## Artifacts

- `report-3x64gib.json`, `report-2x96gib.json`, `report-2x80gib.json`,
  `report-2x75gib-nofit.json` — full planner reports (per-stage retained
  weights, conversion overlap, cache reservation, allowances, utilization).
- `host-plan-2x96gib.json` — the deployable 2-host ranges plan.

## Next under #58

Promoted/packed stage construction from the NAS source into a host scratch
copy per this plan (read-only NAS, staged host residency), then first
guarded executor cells against the real weights.
