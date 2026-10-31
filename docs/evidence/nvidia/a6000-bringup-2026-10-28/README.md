# RTX A6000 bring-up — desktop GPU swap (2026-10-28)

The desktop's RTX 3090 was physically replaced with an **RTX A6000**
(UUID `GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed`, 44.47 GiB, driver
615.71.09). Nothing was reinstalled — the existing venv
(`/home/damen/3090-damendaemon`, torch 2.13.0+cu130 / NCCL 2.29.7)
enumerates and serves on it out of the box. Historical 3090 measurements
elsewhere in these docs remain records of what ran then; live fleet
topology notes now carry the A6000.

## What was verified on the new card

1. **`damendaemon preflight`** (report: `preflight-a6000.json`): torch
   2.13.0+cu130, CUDA 13.0, NCCL 2.29.7, A6000 enumerated with UUID and
   44.21 GiB free, MIG unsupported, all items pass.
2. **Cross-host supervised worker** (the #608 final cell, see
   `docs/evidence/distributed/supervised-live-608/`): the A6000 was the
   SSH-spawned rank-1 worker for three full pair generations — model load,
   layer stage, cross-host NCCL collectives with the A2, completions
   served, killed and respawned per generation.
3. **Single-card serve, 0.8B Q8 smoke** (loopback `serve`, batch 1):
   `triton-hybrid+q8_0`, 348 tok/s decode single-slot, 1665 MiB.
4. **Single-card 27B Q8 — first in the fleet on one GPU**
   (report: `27b-completion.json`): Qwen3.8-27B @ `1d4bf0f2` (52 GiB BF16
   snapshot) quantized q8_0 on the A6000 under `serve --host-gib 40`
   (the default 8 GiB host budget honestly rejects the ~29 GiB estimate
   at registration — `model exceeds configured context or host capacity`;
   raise the budget deliberately). Measured, single completion, one
   warm follow-up, batch 1, ctx 1024, temperature 0:
   - Cold activation (52 GiB disk read + quantize): 61.9s; first token
     62.9s.
   - Warm: first token 1.17s, **decode 14.7 tok/s single-slot**,
     resident 27.09 GiB of 44.47 (17 GiB headroom remains).
   - Correct greedy text (" Paris." / " 100°C at sea level. At higher
     alt…").

## Honest scope

These are capability probes on the new hardware (admission, numerics at
greedy, memory, timing), not SLO or long-context qualification of 27B —
that remains the open qualification work tracked under #60/#68/#3. The
serve instance was stopped and the card returned to idle after the
probe.
