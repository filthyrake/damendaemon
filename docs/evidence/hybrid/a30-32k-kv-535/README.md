# #535 gate 2: the #70 32K A30 contract retried with Q8_0 KV — PASSED

## Contract

The exact #70 capacity cell: pinned Qwen3.8-27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
Q4_0 weights, compensated precision, context 32768, input 32256 raw token ids
(prompt digest `31836e96…be727`), output 512, C1 batch 1, prefill chunk 127, no
graphs, single-device A30 `GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a` under
`guard_shared_cuda`. Retried with `DAMENDAEMON_HYBRID_KV=q8_0` (#535 gate 1
machinery). No context shortening, no chunk change, no batch change.

## Attempt history (recorded honestly)

- Attempt 1 (guard default reserve 2560 MiB): the guard's GPU reserve gate
  stopped the run at 693 s during the 32K request ramp (last observed free
  2756 MiB). This is the same class of guard-kill that #70's attempt 3 hit;
  the #70 attempt-5 configuration (reserve 0, thermal/identity/host gates
  intact) is the established harness for this cell.
- Attempt 2 (`--gpu-reserve-mib 0`, the exact #70 attempt-5 configuration):
  **completed**, 1316.2 s guard wall, peak 53 °C, cleanup complete, GPU
  released clean.

## Result (`q32k-kv-gate.json`, sha256 9517ec75…)

| measure | value | #70 (FP16 KV) |
| --- | --- | --- |
| gate | **completed: true** | failed (torch OOM, 260 MiB ask vs 259 free) |
| first streamed output | 230.3 s (limit 600) | ~120.7 s prefill then OOM |
| total request | 629.0 s (limit 1200) | — |
| usage | prompt 32256, completion 512/512 | — |
| resident KV (status) | **1.062 GiB (q8_0)** | ~4.29 GiB (FP16) |
| cuda peak allocated | 20.182 GiB | 21.97 GiB at OOM |
| warmup activation | 660 s (first_output on 512-token warmup) | ~591 s (#497) |

The Q8_0 KV cache cut the resident full-attention KV from ~4.29 GiB to
1.062 GiB (verified via the `kv_cache_mode`/`kv_cache_bytes` status fields from
the #540 accounting), a ~3.2 GiB saving against the 260 MiB shortfall that
failed #70. Peak allocated 20.182 GiB fits inside the 24 GiB card with the
guard's thermal/identity/host gates intact.

Output preview is fluent continuation of the pinned repetition prompt. Capacity
only: this is not quality, SLO, or performance qualification — decode runs the
eager full-attention path (quantized KV bypasses the fused dense attention
kernel, which reads the FP32 layout), so no throughput claims are made.

## Honest accounting

- Two attempts consumed the one retry cycle: attempt 1 was a guard-kill
  (reserve misconfiguration for this cell, not a capacity result), attempt 2
  is the definitive run. No OOM occurred in either attempt; the allocator
  evidence above is from the passing run.
- The quantized KV is storage-compressed only — values are dequantized before
  attention. Nothing here revises the #70 FP16-KV failure; the cell passes
  because the KV component shrank.
- Production context admission is unchanged: the run used the #68/#70 context
  probe (`hybrid_q4_context_probe`), which substitutes context admission above
  512 for this bounded experiment only.

## Files

- `q32k-kv-gate.json` — full gate report (rounds, statuses, digest).
- `q32k-kv-guard.json` — guard report (attempt 2: phase completed).
- `q32k-kv-serve.log` — server log.
