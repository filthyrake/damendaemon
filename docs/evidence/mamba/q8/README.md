# Mamba CUDA Q8 qualification

Same pinned 130M/370M checkpoints as the parent evidence, A2 GPU, context 512,
32 generated tokens and identical input IDs; five warm HTTP rounds. Q8_0 packs
aligned projection/output matrices. Input embeddings, convolution, normalization,
SSM parameters and activations remain FP32. The 130M rank-48 time projection
remains FP32; the 370M rank-64 projection is packed. Tied output weights get a
separate packed copy. Metal Q8 and Mamba Q4 are explicitly unsupported.

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
