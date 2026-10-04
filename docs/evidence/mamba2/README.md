# Independent Mamba-2 qualification

Issue #31. Published checkpoint `state-spaces/mamba2-130m`, revision
`3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49`:
https://huggingface.co/state-spaces/mamba2-130m

The original checkpoint uses a non-HF configuration and a PyTorch state dict.
`scripts/prepare_mamba2.py ORIGINAL OUTPUT --tokenizer MAMBA_130M_HF` performs an
offline weights-only load, strict reference state-dictionary validation and
exact tensor comparison before writing FP32 safetensors. The source FP16 values
are exactly representable in FP32. The singular embedding key becomes the HF
plural spelling; tied tensors are deduplicated. The preparation report records
the original file hash. Tokenizer comes from the pinned Mamba-130m-HF checkpoint
used in the Mamba-1 evidence. Original weights are downloaded separately.

Configuration mapping/reference: Hugging Face Transformers Mamba2 implementation
and its Apache-2.0 converter (state-spaces/mamba2 and Hugging Face contributors):
https://github.com/huggingface/transformers/tree/main/src/transformers/models/mamba2
Production generation uses dllm's loader, native CUDA/Metal recurrence, shared
native convolution, gated RMS normalization, projections and scheduler. It does
not use a Transformers model or mamba-ssm runtime. The conversion utility is an
offline preparation step, not an inference dependency.

Initial scope: pure single-group Mamba-2 with FP32 weights/activations/state,
context up to 2048, raw completions, graph decode on CUDA, slot isolation,
model residency, optional chunked prefill and complete-prefix snapshots.
Multi-group variants, Codestral, Falcon-H1 hybrids, quantized execution,
speculative decoding and longer contexts are not qualified. No fabricated chat
template is supplied. Registration requires the prepared safetensors directory.

Qualification:
- 16 native recurrence cases per GPU against independent CPU equations, spanning
  state widths 1/33/128/256, odd head widths, reset/continuation and reordered slots.
- 139 independent CPU FP32 logits checks per GPU, plus recurrent/convolution
  state and exact greedy generation, through full 2048 context. Maximum relative
  RMS: A2 0.00002930; M4 0.00002426. CPU reference is Transformers 4.51.3 with SSD
  chunk size 16 to bound its temporary memory; dllm uses its own recurrence.
- Twelve full-recomputation checks per GPU for chunking and prefix snapshots,
  including interleaved slots, suffix continuation, no rewind and invalid states.
- Real HTTP streaming, concurrent requests, cancellation, context bounds and
  Mamba-1/Mamba-2 model switching with host restoration, on both GPUs with chunking
  and prefix snapshots enabled.
- All 219 runtime tensors agree bitwise with the dequantized FP32 GGUF comparator,
  including tied output weights. Prepared/preflight bytes both equal 515,958,528.

## Initial serving measurements

Same source weights, FP32, exact shared input IDs, 512 context/slot, 32 output
tokens and five warm rounds. llama.cpp pinned commit remains
`836d57176dc699a726c55418e4f96b8ca628e1bf`. Every measured output text agrees.

| Device / engine | c1 tokens/s | c2 | c4 | First text ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| A2 dllm | 224.76 | 293.30 | 508.05 | 20.50 / 22.85 / 45.00 |
| A2 llama.cpp | 149.07 | 206.48 | 301.26 | 32.69 / 53.12 / 100.79 |
| M4 dllm | 39.37 | 78.37 | 201.08 | 34.68 / 47.09 / 60.17 |
| M4 llama.cpp | 79.43 | 85.41 | 114.85 | 33.00 / 69.88 / 85.07 |

Apple single-request performance remains behind; laptop trials vary materially.
The host profile distinguishes dispatch/readback durations but does not equate
GPU waits inside a host call with pure Python overhead. These are narrow initial
measurements, not broad parity claims. vLLM comparison is being measured separately.
