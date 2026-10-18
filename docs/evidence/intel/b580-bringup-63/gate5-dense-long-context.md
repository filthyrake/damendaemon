# #63 gate 5: dense Q8 long context beyond 2048 on XPU (passed, 2026-10-18)

## Implementation (PR #528)

The XPU bring-up context cap (2048) now admits dense Q8_0 execution to 16384:

- `xpu_policy.context_limit`: 16384 for `q8_0`, 2048 otherwise. FP32 weights keep
  the original bring-up limit; Q4 stays rejected; hybrid Q8 stays bounded by its
  geometry policy (context <=1024). Dense Q8 leaves FP32-KV headroom: the 0.6B
  dense fixture needs ~0.22 MB/token (FP32 KV), so 16K batch-1 fits beside the
  2 GiB display reserve.
- `validate_limits(context, batch, quantization)` threads the policy through
  `validate_model`; `Runtime` validates the context against each served model's
  quantization policy (manifest-aware), keeping rejection before `require_gpu`;
  the CLI pre-check takes the strictest limit across the served manifest policies.
- Dense fixture staged on the host: Qwen3-0.6B safetensors, pinned revision
  `c1899de289a04d12100db370d81485cdf75e47ca` (the same snapshot #2's evidence
  used), downloaded to `/home/damen/models/Qwen3-0.6B-c1899de`
  (model.safetensors sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`).

## Gate result (rocky10-desktop, B580, torch 2.10.0+xpu)

Runner `scripts/experiments/xpu_dense_context_gate.py`, two served rounds:

- **Long context (16384 context, batch 1, Q8_0)**: 5081-token needle-retrieval
  prompt — the model returned `Answer: ZEPHYR-7391` (the buried access code,
  correct), **byte-identical across two requests** (greedy determinism).
  Prefill+decode 6.4 s cold / 2.8 s warm (capability only, never performance).
- **Concurrency (4096 context, batch 4, Q8_0)**: four distinct prompts generated
  concurrently, all producing tokens, XPU steady state 5.02 GB / peak 6.29 GB.
- **Memory at 16K**: steady state 5.03 GB allocated, peak 7.27 GB — inside 12 GiB
  with the 2 GiB display reserve active on the live desktop.

## Scope and honesty notes

- This is a **capability qualification** (retrieval + determinism + concurrency +
  memory), not an independent-reference logit gate; no golden-trace RMS comparison
  was run for dense Qwen3 Q8 long context.
- Concurrent-round outputs are raw-completion continuations from a small dense
  model (rambling style); the gate asserts correct token production and stable
  serving, not answer quality.
- The hybrid fixture remains bounded to context 1024 under its geometry policy;
  FP32 long context is not admitted.

## Files

- `xpu-longctx-gate.json` — gate report (rounds, memory, needle, determinism)
- `xpu-longctx-gate.serve-long.log` / `xpu-longctx-gate.serve-concurrent.log` — serve logs
- Manifests retained on the host next to the logs (not committed)
