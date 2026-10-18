# #63 Intel Arc B580 bring-up — gate 1: device enumeration + real compute (passed, 2026-10-18)

## Result

The B580 runs real torch-XPU compute on rocky10-desktop. The device probe
(`scripts/experiments/xpu_device_probe.py`) completed with:

- Device: `Intel(R) Arc(TM) B580 Graphics`, platform "Intel(R) oneAPI Unified
  Runtime over Level-Zero V2", GPU driver `1.15.38646+6` (NEO
  `intel-level-zero-26.22.38646.6-2.el10_2`), 12,809,404,416 bytes total
  memory, UUID `86800be2-0000-0000-8400-000000000000`.
- Compute proof: FP32 64×64 matmul **exact vs CPU control** (max abs 0.0),
  FP16 host round-trip exact, FP16 matmul executes, allocator counters live.
- Stack: Rocky 10 (kernel 6.12 CIQ), Python 3.12.14, isolated venv at
  `/home/damen/b580-dev/venv`, **torch 2.10.0+xpu** (pip xpu index; bundles
  the 2025.3.1 Intel runtime).

## Root cause of the initial segfaults (the real work of this gate)

Every torch-xpu version (2.14.1 and 2.10.0) segfaulted with **ip=0 (NULL
function-pointer dispatch)** on the first device allocation, while
enumeration succeeded. Findings:

- The system Level Zero **loader** is `level-zero-1.21.9.0-1142.el9_5` — an
  **EL9 build on an EL10 host**, and the newest the `intel-graphics` repo
  offers. The GPU driver itself (`intel-level-zero-26.22.38646.6-2.el10_2`)
  is a current EPEL el10_2 build.
- With the system loader, `torch.xpu.is_available()`/`device_count()` work
  but device allocation crashes (kernel log: `segfault at 0 ip
  0000000000000000 error 14`, reproducible across torch versions).
- Building the **upstream loader** (oneapi-src/level-zero, tag v1.34.0) from
  source into `/home/damen/b580-dev/ze-loader` (local prefix, no system
  change) and prepending it via `LD_LIBRARY_PATH` fixes everything: full
  compute works on both torch 2.10 and (by implication of the same dispatch
  path) the loader is the only broken component.

Environment note: sourcing `/opt/intel/oneapi/setvars.sh` is NOT needed for
torch (the 2026-10-06 enumeration fix applied to the system SYCL tools) and
mixing the system oneAPI libs with torch's bundled runtime is harmful — run
torch with only the local loader prepended.

## Reusable bring-up recipe (desktop)

```
export LD_LIBRARY_PATH=/home/damen/b580-dev/ze-loader/lib64:$LD_LIBRARY_PATH
cd /tmp/damendaemon-63 && /home/damen/b580-dev/venv/bin/python -B -m \
  scripts.experiments.xpu_device_probe --output <new-path>.json
```

The loader build lives in `/home/damen/b580-dev/level-zero` (git tag v1.34.0,
installed to `/home/damen/b580-dev/ze-loader`). Rebuilding is ~1 minute; no
system packages were changed, upgraded or removed.

## Scope

Gate 1 of #63 only: read-only inventory, runtime selection, device identity,
real-device compute. No model load, no display-headroom load measurement, no
performance claims. Remaining gates: model fit + quantization on XPU,
context/scheduler lifecycle, concurrency, matched baseline.

## Files

- `b580-probe-final2.json` — passing probe report
- `segfault-probe-attempt.json` / `segfault-stepwise.log` — retained
  pre-fix failure evidence (system loader)
