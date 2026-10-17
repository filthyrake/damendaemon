# Mamba-2 Q8 and native Metal Q8 projections

The same pinned 130M checkpoint now supports Q8_0 input/output projections and
output head, with unchanged FP32 embeddings, convolution, norms, discretization
parameters and recurrent state. Prepared weights shrink from 515,958,528 to
292,337,664 bytes, including scales and a separate packed tied output head.
All 49 packed matrices are included in exact preflight accounting. The Q8 GGUF
comparison verifies all 219 dequantized tensors bitwise against damendaemon weights.

Both CUDA and Metal pass 139 independent quantized-reference logits checks,
state checks and exact greedy generation through 2048 context. Both also pass
chunk/prefix reference checks and real HTTP/cancellation/hot switching with
Mamba-1. Metal's new packed projection uses FP32 activations/accumulation and
shares weights across up to four decode rows. Longer prompts use a temporary
FP32 expansion for matrix multiplication, with no resident expanded weight copy.
The 14-case Metal projection oracle uses independently constructed signed Q8
integers/scales and CPU FP64, odd output widths, bias, zero scales, prompt rows
and inputs above FP16 range. See the Mamba-1 Q8 Metal evidence for that oracle.

Same context512, input IDs and 32 outputs as the FP32 comparison; five warm rounds:

| GPU / engine | c1 tokens/s | c2 | c4 | First text ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| A2 damendaemon Q8 | 420.67 | 415.84 | 589.72 | 22.42 / 48.36 / 52.31 |
| A2 llama.cpp Q8 | 287.33 | 302.13 | 339.09 | 25.15 / 52.63 / 93.91 |
| M4 damendaemon Q8 | 64.11 | 117.26 | 229.84 | 29.18 / 39.74 / 65.40 |
| M4 llama.cpp Q8 | 152.72 | 114.81 | 123.00 | 21.89 / 47.12 / 88.57 |

damendaemon text is stable across these concurrency levels and both GPUs. All Apple
comparison text agrees. **CUDA llama.cpp Q8 output differs**, including among
its own trials/batch sizes. Matching stored tensors does not mean matching
backend arithmetic; the cause has not been isolated. These are not cross-engine
CUDA output-equivalence claims. Apple single-request performance remains behind,
and laptop timing variation prevents claiming a stable universal speedup.

Quantization also changes the model versus FP32. The separate CPU drift study
uses ten published authored passages (192 predicted tokens): perplexity rises
24.20023 to 25.42213, and teacher-forced top-1 agreement is 85.42%. This tiny
corpus is not a broad quality evaluation, but disproves a claim of unchanged
predictions. Q8 is opt-in. No Mamba-2 Q4 or multi-group/hybrid claim is made.
