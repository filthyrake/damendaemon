# #63 gate 2: first model generation on the B580 (passed, 2026-10-18)

## Result

The runtime's pushed XPU adapter paths (cf80704/ffb30e6) generated text on
real Intel Arc B580 hardware for the first time:

- Command: `damendaemon run <Qwen3.5-0.8B snapshot> --raw --max-tokens 24`
  with `DAMENDAEMON_DEVICE=xpu DAMENDAEMON_EXPERIMENTAL_XPU=1
  ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and the local Level Zero loader
  prepended (see gate 1 README).
- Model: pinned Qwen/Qwen3.5-0.8B revision
  `2fc06364715b967f1860aea9cf38778875588b17` (1.7 GB, pulled via
  `damendaemon pull` to the standard HF cache).
- Output: `17 * 23 = 391.` — correct, greedy (`temperature` default 0). Two
  independent runs byte-identical (`xpu-greedy-run-a.log` /
  `xpu-greedy-run-b.log`).
- Backend recorded by the runtime: `kernel_backend: "torch-hybrid+float32"`,
  `weight_quantization: none` — FP32 eager hybrid text path.
- Runtime self-timing (single stream on the shared desktop GPU, recorded as
  capability evidence only, NOT a performance claim): activation 1.22 s,
  first token 2.31 s, 24 sampled tokens at ~23.5 tokens/s sequential decode,
  total 3.29 s for 15 prompt + 24 completion tokens.

## Required environment (recorded verbatim from the runtime's own gates)

1. `DAMENDAEMON_DEVICE=xpu`
2. `DAMENDAEMON_EXPERIMENTAL_XPU=1` — the runtime refuses XPU without this
   explicit opt-in ("execution is unqualified")
3. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` — required after the opt-in
4. `LD_LIBRARY_PATH=/home/damen/b580-dev/ze-loader/lib64:$LD_LIBRARY_PATH` —
   the locally-built upstream Level Zero loader v1.34.0 (gate 1 root cause)

Dependency stack in the venv: torch 2.10.0+xpu, safetensors 0.5.3,
tokenizers 0.21.1, huggingface-hub 0.30.2, jinja2, einops 0.8.1
(`fla-core` was NOT installed; the eager FP32 hybrid path did not require
it — fused/FLA-backed paths remain untested and must not be assumed).

## Scope

Gate 2 only: one small hybrid dev fixture at 24 tokens, FP32, single stream.
Not quantization, not context/scheduler lifecycle, not concurrency, not the
32K product target. Raw single-run timings on a Wi-Fi-connected desktop the
user was actively using; never quote them as performance evidence.

## Files

- `xpu-greedy-run-a.log` / `xpu-greedy-run-b.log` — full CLI output including
  usage and self-timing JSON
- `b580-probe-final2.json` — gate 1 device probe report (same commit)
