# Standalone GGUF import

Initial checkpoint qualification: Qwen/Qwen2.5-0.5B-Instruct, pinned HF revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Q8_0 comparator file on NVIDIA A2,
PyTorch 2.6.0+cu124. Import uses the optional MIT-licensed `gguf==0.19.0` reader
package with additional bounds checks. Inference stays in dllm's decoder and
GPU kernels. No llama.cpp inference dependency is introduced.

Fixture SHA-256: `6810aa35d0eae4a02dea4031bcb983842888739028ebfdce797e16bf5b869d18`.
Reference llama-server executable SHA-256:
`344fe55b9043e7cd362b199a24fcbb9eba7e1d9513e17083a18df4190002401d`.

## Proof

- [A2 runtime report](qwen2-q8-a2.json): all 171 runtime weight tensors and 97
  scale tensors exactly match dllm's existing Q8 preparation from the pinned
  HF weights. Packed values and scales are copied, not requantized.
- Complete logits match bit-for-bit for eight generation steps at 1, 17, 129,
  and 505 prompt tokens, both with and without CUDA graphs (64 comparisons).
- Actual HTTP registration, chat, streaming, concurrent requests, cancellation
  recovery, host restoration after a model swap, and complete unload pass.
- Four alternating warm HTTP rounds, identical eight-token input and 16-token
  output: GGUF median 90.95 ms, HF-prepared Q8 median 93.94 ms. This is a small
  regression check showing no observed import-path penalty, **not evidence of
  a kernel speedup or a comparison with another engine**. Model activation and
  graph warmup are excluded.
- [Independent tokenizer report](qwen2-tokenizer.json): all 15 cases match
  llama.cpp's `/tokenize` with `parse_special=true` and `add_special=false`,
  including multilingual text, Unicode combining marks, emoji, whitespace,
  code, JSON/tool tags and control characters.
- The combining-mark and accented-text cases intentionally differ from the HF
  tokenizer's NFC-normalized output. GGUF follows the embedded tokenizer's
  behavior; it does not invent a normalizer absent from GGUF metadata.
- 257 CPU regression tests pass. Synthetic import tests cover F16/F32, Q4_0,
  Q8_0, Qwen3 Q/K norms, packed embeddings, tied heads, file-cache invalidation,
  incompatible fused formats, truncated/overlapping tensors, oversized metadata
  arrays, and non-finite packed scales.

Reproduce using `scripts/verify_gguf_tokenizer.py` and
`scripts/verify_gguf_runtime.py`. The latter requires a value-identical comparator
exported with `scripts/make_q8_comparator.py`; it does not assert equivalence
between unrelated quantizations.

## Boundaries

The initial importer accepts dense Qwen2/Qwen3 and little-endian GGUF v2/v3 with
F16/F32 or Q4_0/Q8_0 tensors. Acceptance is broader than this first A2 checkpoint
qualification. K-quants, IQ formats, split checkpoints, other architectures,
scaled RoPE, vision/audio adapters and arbitrary GGUF tokenizer families are
not implemented. Q/K/V and gate/up fused components must each use compatible
formats; distinct projections may use different supported formats.

Floating weights follow the existing FP16 decoder-weight policy, including
F32 vectors. GGUF import is not an all-FP32 execution mode. Packed embeddings
expand to FP16 for lookup in bounded row chunks; a tied packed output head
retains its original packed storage. The metadata cache holds at most two
plans keyed by file identity, never mmap weight views. File changes invalidate
plans, and final host storage must match the preflight byte estimate.

Q4 is CUDA-only. Apple Silicon and Qwen3 GGUF runtime qualification are separate
work; existing native kernel qualification alone is not an import proof.
Broader import work is tracked in GitHub issue #43.
