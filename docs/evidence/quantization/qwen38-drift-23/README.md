# Qwen3.8-27B original-weight Q4_0 drift study (#23, 2026-10-18)

## Result

`scripts/quantization_drift.py` on the practical checkpoint — Qwen3.8-27B,
pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (safetensors,
downloaded to the desktop host) — original weights versus independent upstream
gguf q4_0 roundtrip:

| Weights | Perplexity (10-text corpus, 190 tokens) |
|---|---|
| Original (FP16 storage) | 10.4536 |
| Q4_0 roundtrip | 10.6178 |

- Teacher-forced top-1 agreement vs original: **87.1%** (165.5/190 tokens)
- Per-text relative logit RMS: mean 9.87%, max 12.52%
- Perplexity drift: +0.164 (~1.6% relative)

## Conditions and honesty notes

- Device: rocky10-desktop CPU, **float16 arithmetic**, 32 threads (the FP32
  CPU study needs ~108 GB resident — unavailable; FP16 arithmetic is declared
  in the report and both passes share it, so the comparison is internally
  consistent).
- Corpus: the script's fixed 10-text authored corpus — measures quantization
  drift, not broad task quality.
- Quantizer: independent upstream gguf q4_0 (gguf-py conventions), not
  damendaemon's packed kernels; the runtime's packed q4_0 execution was
  already qualified against a Q4 oracle (84/84 on HOMER and A30). This study
  adds the original-weight dimension that the oracle match could not provide.
- Corpus-level top-1 agreement of 87.1% is real drift: greedy text from the
  Q4 model will diverge from the original on roughly one token in eight under
  teacher forcing. This is recorded as-is; no quality threshold is claimed.

## Scope

This is the #23 original-weight quantization-drift/quality evidence cell on
the practical checkpoint. It is not a competitive-quality benchmark, not a
human evaluation, and not a performance measurement.

## Files

- `drift-qwen38-q4.json` — full report (both passes, per-text RMS)
