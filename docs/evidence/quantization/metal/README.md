# Native Metal Q8 for dense attention decoders

The existing signed-byte/FP16-scale Metal projection now serves Qwen2, Qwen3
and Llama decoders. Activations, accumulation and KV remain FP32. Small batches
read packed weights directly; prompt matrices temporarily expand one projection
at a time. There is no persistent dequantized model copy. Q4, Qwen3.5 hybrid,
Gemma and other unqualified Metal quantization combinations still fail clearly.
Q8 remains opt-in (`DLLM_QUANTIZATION=q8_0` or per-model registration).

The pinned Qwen2.5-0.5B-Instruct, Qwen3-0.6B and Llama-3.2-1B checkpoints used
by the architecture qualification pass independent CPU FP32/gguf-Q8 reference
traces through 2048 tokens, including batch reorder, reset and reused slots:

| Checkpoint | Reference checks | Maximum relative RMS logit error |
|---|---:|---:|
| Qwen2.5-0.5B | 82 | 0.00001581 |
| Qwen3-0.6B | 84 | 0.00001508 |
| Llama-3.2-1B | 82 | 0.00001003 |

Qwen3 passes 48 chunk/prefix-cache reference checks. Real HTTP covers all three
models with one and three resident models, cold/host/GPU restoration, concurrent
requests, streaming, template controls and cancellation. Qwen2 also passes
independent token-score checks and streaming/nonstreaming scored API tests.
Reference agreement with quantized weights is distinct from original-checkpoint
quality; the existing Qwen2 Q8 drift evidence remains applicable.

## Serving measurements

Apple M4, context 512, the same 16 input IDs, 32 outputs and five warm rounds.
FP16 denotes retained weight storage; dllm activations/KV are still FP32.
The Q8 comparator's 291 tensors match dllm's dequantized values bitwise.
The comparator rounds retained vectors to dllm's prepared FP16 values before
storing them as FP32, as required by llama.cpp's vector kernels. The earlier
source vectors differed slightly; storing vectors directly in FP16 also failed
upstream execution. Neither failed comparison is used for throughput claims.

| Engine / storage | c1 tokens/s | c2 | c4 | First text ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| dllm Q8 | 48.90 | 83.97 | 148.86 | 50.64 / 85.51 / 125.11 |
| llama.cpp Q8 | 128.76 | 211.03 | 259.64 | 18.19 / 21.50 / 58.80 |
| dllm FP16 | 60.11 | 101.69 | 178.61 | 51.35 / 87.46 / 125.16 |

Q8 is slower here than dllm FP16, and dllm trails llama.cpp. This is added
lower-memory coverage, not a performance improvement. Apple timing varies;
these are the recorded runs, not a claim about every workload. Each engine's
text is stable across concurrency, but dllm and llama.cpp produce different text
despite matching stored values. Their arithmetic is not claimed equivalent.

Pinned revisions:
- Qwen2.5: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Qwen3: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Llama3.2: `5a8abab4a5d6f164389b1079fb721cfab8d7126c`.
- llama.cpp: `836d57176dc699a726c55418e4f96b8ca628e1bf`.

Qwen2 resident weight/scale accounting is 988,208,640 bytes for FP16 versus
797,389,312 for Q8 (97 packed projections), a 190,819,328-byte reduction. The
unchanged input embedding and separately packed tied output head limit the
saving. These figures exclude KV, temporary projection expansion and allocator
reserve; preflight host estimates also match actual prepared host tensors.
