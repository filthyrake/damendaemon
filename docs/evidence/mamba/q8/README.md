# Mamba CUDA Q8 qualification

Same pinned 130M/370M checkpoints as the parent evidence, A2 GPU, context 512,
32 generated tokens and identical input IDs; five warm HTTP rounds. Q8_0 packs
aligned projection/output matrices. Input embeddings, convolution, normalization,
SSM parameters and activations remain FP32. The 130M rank-48 time projection
remains FP32; the 370M rank-64 projection is packed. Tied output weights get a
separate packed copy. Mamba Q4 remains unsupported. The later Metal qualification is recorded below.

Both checkpoints pass 139 reference logits checks plus state checks through
2048 tokens against CPU Transformers with independently gguf-py-quantized weights.
Maximum relative RMS is 0.00002637 (130M), 0.00001233 (370M); greedy reference
traces agree. This tests implementation accuracy for the quantized model, not
unchanged quality versus the original FP32 checkpoint.

Prepared bytes, including scales, exactly match the preflight estimates:

| Model | FP32 bytes | Q8 bytes | Packed matrices |
|---|---:|---:|---:|
| 130M | 516541440 | 299411328 | 73 |
| 370M | 1486065664 | 607472128 | 193 |

`make_q8_comparator.py` creates benchmark GGUF files with the same tensor policy.
`verify_mamba_gguf.py` independently dequantizes every GGUF tensor and confirms
bitwise equality with dllm's prepared weights (243/483 tensors). This verifies
stored weights, not identical arithmetic across engines. **Greedy text differs
between dllm and llama.cpp**, and llama.cpp's 130M output also varies with batch
size. The cause is not isolated. These measurements are not output-equivalence
proof or an assertion that either quantized engine preserves FP32 output.

| Model / engine | c1 tokens/s | c2 | c4 | TTFT ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| 130M dllm | 405.27 | 413.50 | 533.08 | 25.86 / 44.82 / 70.31 |
| 130M llama.cpp | 223.69 | 301.47 | 314.26 | 33.54 / 38.42 / 84.31 |
| 370M dllm | 176.26 | 213.27 | 306.65 | 66.24 / 90.44 / 102.69 |
| 370M llama.cpp | 103.24 | 125.98 | 155.82 | 58.60 / 79.56 / 103.71 |

Raw commands, samples, timing trials and tensor/storage checks accompany this
file. Real HTTP checks cover streaming, concurrent requests, cancellation,
context boundaries and 130M/370M switching, including restoration from host RAM.
The A30 was not used. No Q8 vLLM comparison was run.

## Native Metal Q8

Mamba-130M and 370M each pass 139 independent quantized-reference logits checks
plus recurrent state through 2048 on M4. The 14-case native projection oracle
uses CPU FP64, signed extreme values, zero scales, bias, odd dimensions, prompt
rows and activations above FP16 range. Mamba-2/Mamba-1 Q8 hot-switching, chunks,
prefix snapshots and real HTTP proof are in the Mamba-2 Q8 evidence.

Matched 130M M4 Q8 context512/32-output/five-round throughput c1/c2/c4:
dllm 58.63/114.09/272.00 tokens/s; llama.cpp 158.09/302.56/333.84. First text:
dllm 28.09/45.84/57.42 ms; llama.cpp 22.80/28.85/62.14 ms. All Apple trial text
agrees and matches dllm CUDA Q8. Apple still trails; these results do not imply
parity. Raw results are under `metal/`.

Separate ten-passage CPU drift study (192 predicted tokens) versus FP32:
130M perplexity 23.76786 -> 24.20444, top-1 agreement 92.19%;
370M perplexity 16.99595 -> 17.14334, agreement 97.40%.
This is a small authored drift probe, not a broad model-quality benchmark.
