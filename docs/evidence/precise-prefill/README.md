# Compensated Tensor Core projections for longer prompts

Related issue #34. The higher-precision dense CUDA policy previously used the
compensated kernel for 3–128 rows, then expanded stored FP16 matrices to FP32
for larger prompts. Larger rows now use the same native Triton kernel: scaled
high/residual FP16 activation components, two Tensor Core products and FP32
accumulation. FP16 matrix storage and FP32 input/output dtypes are preserved.
This remains a qualified approximation, not bitwise FP32 arithmetic.

The change applies to explicit higher-precision Qwen3/Llama and the existing
YaRN policy. Default standard/mixed projections and Metal are unaffected.
Gemma has its own projection implementation. No draft model, narrowed output
vocabulary, quantization change or reference-tolerance relaxation is involved.

## Projection measurements

A2, five synchronized trials after two warmups, FP32 inputs/output and FP16
stored weights. The old baseline includes its real weight conversion cost.
The probe covers 129/512/2048 rows across three Llama-shaped matrices. The
compensated path is 1.67–2.83× faster, with maximum absolute errors below
0.000021 and relative RMS recorded in `probe-large-precise-matmul-a2.json`.

This is a kernel measurement, not an end-to-end claim. Temporary activation
buffers also cost memory: at 2048 rows on the 8192→2048 projection, peak extra
allocation is essentially unchanged at 80 MiB. Other tested shapes save memory,
but reduced matrix-conversion storage is not a universal peak-memory guarantee.

## Independent precision gates

Existing FP32 reference traces pass all 332 comparisons across Qwen3-0.6B and
1.7B (84 each, context 2048), Llama3.2-1B (82, context 8192), and the YaRN Qwen2
fixture (82, context 2048). They include templates, long/padded-boundary prompts,
rolling slots and decoding after slot reuse. Every independent greedy argmax
matches. Maximum absolute logit errors are 0.000298, 0.000675, 0.000151 and
0.000279 respectively. Raw reports retain checkpoint paths/revisions and policies.

## Complete HTTP A/B

Same resident model, explicit higher-precision policy and FP16 stored matrices;
four alternating original/optimized rounds after warmup. Both modes retain the
new fused FP32 attention. Speculation and prefill graphs are disabled. The
baseline restores FP32 matrix expansion only above 128 rows. Every response's
text, finish reason and usage matches; all requests emit 32 tokens.

| Model | Prompt tokens | Original complete ms | Optimized complete ms | Original first-token ms | Optimized first-token ms |
|---|---:|---:|---:|---:|---:|
| Llama3.2-1B | 512 | 1051.687 | 825.077 | 449.840 | 222.675 |
| Llama3.2-1B | 2048 | 2513.303 | 1775.718 | 1859.611 | 1123.303 |
| Llama3.2-1B | 8160 | 10602.360 | 7274.680 | 9753.592 | 6426.979 |
| Qwen3-0.6B | 512 | 578.28 | 469.89 | 229.37 | 120.36 |
| Qwen3-0.6B | 2048 | 1597.78 | 1286.24 | 1017.69 | 710.17 |
| Qwen3-1.7B | 512 | 1428.91 | 1076.84 | 674.53 | 325.06 |
| Qwen3-1.7B | 2048 | 3634.11 | 2599.63 | 2655.79 | 1623.59 |

Complete times come from the HTTP client; first-token intervals are recorded by
the server, not a streaming client's network TTFT. These are warm measurements
on narrow authored workloads, not broad benchmark parity. Llama reserves context
8192; Qwen3 reserves 4096. Cold compilation/model loading is excluded.

Reproduce with `probe_large_precise_matmul.py`, `replay_reference.py` and
`benchmark_fp32_prefill_http.py --optimization projection`. The latter can select
context and prompt lengths explicitly, and retains the attention A/B mode for
reproducing the preceding optimization.

## Boundary and verifier integration

Qwen3-0.6B and Llama additionally pass the independent traces with 257-token
prefill chunks (84/82 checks), exercising the new projection path and nonzero
attention offsets together. That brings independent model comparisons to 498.
All three models pass 18 verifier width/slot/acceptance cases after a 257-token
raw prefix, including forced rejection, isolation of another KV slot and graph
reclamation. Reports explicitly identify FP32 compute/KV/residuals and FP16
matrix storage. The CPU/HTTP regression suite passes 216 tests.
