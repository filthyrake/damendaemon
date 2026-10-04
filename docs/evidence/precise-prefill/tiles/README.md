# Larger projection tiles, with a measured cutoff

Related issue #34. At 512 or more rows, compensated projections use 64×128×64
matrix tiles, eight warps and two pipeline stages. Smaller shapes retain the
former 64-column layout. The split/scale/accumulation algorithm is unchanged.
`wide_tiles=False` preserves the former launch shape for controlled comparison.

The A2 sweep evaluates eight configurations across 12 shapes. The selected layout
improves larger rows by up to 43%; held-out dimensions at 777/4096 rows improve
17–45%. One 257-row held-out shape regresses by 13%, so the runtime retains its
old layout below 512. All sampled outputs of the selected layout match the
existing compensated kernel exactly; this does not prove universal bitwise
invariance or equivalence to full FP32 arithmetic.

A small A30 probe checks the same candidate while its existing server is idle.
It uses modest temporary allocations and releases them on completion; the
existing server retains its 13411 MiB allocation. Most larger shapes improve
about 5–15%, while some small cases are essentially flat. This is kernel-only
A30 evidence, not full-model A30 qualification or a cross-GPU speed comparison.

## Model and HTTP gates on A2

416 independent comparisons pass: Qwen3-0.6B/1.7B, Llama3.2-1B through context
8192, YaRN Qwen2, and a 512-token Qwen3 chunk boundary. Every reference argmax
matches. The full regression suite passes 216 tests.

Same resident explicit higher-precision model, four alternating HTTP rounds,
32 output tokens, fused attention and compensated projections in both modes.
Only the large-row launch shape changes. Every paired response's text, finish
reason and usage matches.

| Model | Prompt tokens | Previous complete ms | Wider-tile complete ms | Previous first-token ms | Wider-tile first-token ms |
|---|---:|---:|---:|---:|---:|
| Llama3.2-1B | 512 | 823.174 | 779.308 | 222.302 | 178.415 |
| Llama3.2-1B | 2048 | 1778.413 | 1520.381 | 1127.853 | 867.427 |
| Llama3.2-1B | 8160 | 7280.894 | 6344.156 | 6426.730 | 5489.797 |
| Qwen3-1.7B | 512 | 1080.269 | 1021.882 | 326.160 | 267.404 |
| Qwen3-1.7B | 2048 | 2603.814 | 2244.680 | 1623.215 | 1262.548 |

Complete timings come from the client; first-token intervals are recorded by the
server. Model load and compilation are excluded. These are narrow warm workloads,
not broad parity claims. Reproduce using `probe_precise_tiles.py` and
`benchmark_fp32_prefill_http.py --optimization tiles`.
