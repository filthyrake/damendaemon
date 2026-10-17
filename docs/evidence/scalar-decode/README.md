# Direct single-request CUDA graph dispatch

`BatchExecutor` captured one-row CUDA graphs, but inherited scalar `decode`
looked only at the deliberately disabled base graph. `Runtime.generate` and
`damendaemon run` consequently issued eager operations. The fix reuses the existing
one-row batch graph and synchronizes scalar length with slot zero. The no-graph
fallback is unchanged; existing Gemma/paged overrides retain their behavior.
The HTTP batching scheduler already used these graphs, so these are **not HTTP
serving speedups**.

## Matched A2 measurements

One resident checkpoint, identical precision and context (512), 32 greedy decode
steps after untimed prefill, six alternating warmed pairs. Reading each next token
synchronizes both paths as in direct generation. Loading, prefill, HTTP transport,
and chat parsing are excluded. All teacher-forced logits and free-running tokens
match exactly in the measured cases; another populated cache slot stays unchanged.
The Qwen3 policy is explicitly FP32, not the standard mixed policy.

| Model | Prompt tokens | Old decode, seconds | Graph decode, seconds | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B FP32 | 5 | 0.799 | 0.265 | 3.02× |
| Qwen3-0.6B FP32 | 64 | 0.762 | 0.270 | 2.82× |
| Qwen3-0.6B FP32 | 256 | 0.772 | 0.306 | 2.52× |
| GPT-OSS-20B MXFP4 / FP32 residual | 5 | 1.991 | 1.948 | 1.02× |
| GPT-OSS-20B MXFP4 / FP32 residual | 64 | 2.000 | 1.958 | 1.02× |
| GPT-OSS-20B MXFP4 / FP32 residual | 256 | 2.012 | 1.971 | 1.02× |

`scalar-qwen3-reference-a2.json` was generated with
`replay_reference.py qwen3-small-fp32-golden.pt --slot 0 --scalar` and
`DAMENDAEMON_QWEN3_PRECISION=float32`. Its 25 independent-reference checks pass at
context 2048, including all greedy IDs; maximum relative RMS error is 5.93e-6.
The report predates the explicit `scalar` report field, which is now emitted by
the script; the invocation above records which path was exercised.

The source script is `scripts/benchmark_scalar_decode.py`. GPU measurements use
only the authorized A2, with no concurrent test workloads. Issue #40 tracks this
bug and its verification; these files are supporting evidence.

GPT-OSS also passes 72 independent streamed FP32-reference checks via
`replay_gpt_oss.py models/gpt-oss-20b gpt-oss-fp32-reference.pt --scalar`:
24 layer outputs in each of two cases, prefix/full logits, scalar teacher-forced
decode, untouched-slot isolation and slot reuse. This reuses the existing
independently generated reference; no new goldens were derived from damendaemon.

All 237 local CPU/HTTP tests pass, including the eager fallback and scalar slot
length contract. The small GPT-OSS gain is consistent with GPU-dominated work;
the Qwen3 improvement should not be generalized to every model or backend.

The actual `damendaemon run` CLI produced 32 tokens from a 15-token raw prompt with the
same pinned Qwen3 model and explicit FP32 policy. Recorded decode rate was
121.14 tokens/s; the 6.13-second cold total includes loading and first-use prefill.
This one cold smoke run is not the source of the matched speedup claim above.
