# AWQ checkpoint format groundwork

GitHub issue #49 tracks import and execution qualification. The format module
is connected to packed execution and normal model dispatch. The initial scope
is CUDA Qwen2 AWQ GEMM with four-bit weights and group size 128. Independent
HF SDPA and runtime layout checks pass; lifecycle and matched engine performance
remain separate gates. Eager-FP16-reference numerical parity is not claimed.

Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct-AWQ`, pinned revision
`db09cd27ead7fee40cdee309693cf83601b9c899`. Its published configuration selects
four-bit AWQ GEMM, group size 128 and asymmetric zero points. The original
checkpoint layout packs output channels, using the shift order
`0,4,1,5,2,6,3,7`; this differs from dllm's existing Q4_0 layout. Zero points
have the same packing and do not use the historical GPTQ plus-one convention.

Format reference inspected: [vLLM AWQ Triton implementation at the local pinned
revision](https://github.com/vllm-project/vllm/blob/84bcbc62644356270aaaa5e2d0237d03adc9bb3a/vllm/model_executor/layers/quantization/awq_triton.py).
The host codec is independently implemented from the format. No vLLM runtime
or Marlin kernel is incorporated by this increment, and the reference checkout
is unchanged.

`qwen25-small-format.json` validates all 168 packed projections, including
38,400 unsigned-word scalar coefficient checks across channel permutations
and input-group boundaries. Every expanded coefficient is finite. The
declared tied output head exactly equals the embedding. The report retains
the checkpoint, codec and verifier hashes. Five unit tests additionally cover
known literal packed words, signed high nibbles, zero-point semantics, group
boundaries, packed fusion and malformed configuration/buffers.

The codec retains the original int32 packed matrices, packed zero points and
FP16 scales. Its explicit dequantizer is an inspection helper, not a resident
dense inference fallback. GPU kernel, independent model, layout, lifecycle
and matched HTTP performance gates remain required before runtime support.

Reproduce with `PYTHONPATH=. python scripts/verify_awq_format.py SNAPSHOT
--output REPORT.json` using the pinned snapshot.

`verify_awq_kernels.py` is the next isolated gate. It loads a pinned upstream
`awq_triton.py` solely as a validation oracle, replacing only its Triton import
in a temporary copy and preserving its copyright header. It records original
and adapted source hashes. dllm's independently implemented packed GPU
products are checked against FP32 accumulation over FP16 coefficients, with
bias/no-bias and three activation magnitudes; coefficient bit patterns must
also match both the host codec and the upstream decoder. Any timings compare
standalone projection paths, not vLLM's production engine or Marlin backend.

The first matrix kernel exceeded A2 per-block shared memory: 208896 bytes
requested versus 101376 available. Its coefficient decoders had already
matched both references. The failure report/log are retained as
`qwen25-small-kernels-shared-memory-failure-a2.*`. The repaired kernel loads
compact packed tiles, then loads group scales and zero points once per tile
instead of replicating those loads across input rows.

`qwen25-small-kernels-compact-a2.json` passes all 378 matrix checks over nine
real/fused projection shapes, rows 1/2/4/33/64/129/256, bias on/off and activation
exponents -10/0/8. All nine decoded projections are bit-exact against both host
and upstream GPU references; worst matrix relative RMS is 0.0414%. Timings are
matrix-only comparisons with expansion plus dense multiplication; rows above
128 already use that path and act as controls.

`qwen25-small-host-import.json` checks the separate strict host loader: 96 fused
packed projections, a shared tied output head, valid chat tokenization and an
exact 458314496-byte storage/admission match. Header preflight rejects missing
or unexpected tensors, wrong dtypes/shapes, mismatched/escaping shard indexes,
unsupported formats and changed sources. Five additional loader tests cover
these failures, tied-head disagreement, nonfinite scales and unsupported
devices/conversions. Runtime registration and full-model GPU validation remain
separate gates.

The independent full-model harness initially failed while assembling HF
parameters from inference-mode tensors. Its second attempt reached decoding
but incorrectly treated the executor's `(logits, token)` return as a tensor.
Both harness failures are retained in the `reference-assembly-failure` and
`reference-interface-failure` reports. Repairs use ordinary no-grad reference
weights with gradients disabled and unpack the existing decode interface;
they do not change model arithmetic or accuracy thresholds.

`qwen25-small-reference-numerical-failure-a2.json` completes all 128 model
checks through context 2048 but passes only 120. Worst relative RMS is 2.704%
and maximum absolute logit error is 0.65235. A separate greedy mismatch occurs
where the independent reference's top two logits tie. Both eager and graph
execution reproduce the failures. Registration remains disabled.

`qwen25-small-reference-expanded-diagnostic-a2.json` replaces packed products
only with temporary expanded FP16 products. It also passes only 120/128,
with 2.683% worst relative RMS and 0.66114 maximum absolute error. This isolates
the failure beyond the packed product alone; it is not a runtime fallback or
a performance result. The verifier exposes explicitly named diagnostic flags
and records them in each report.

The rounded-attention diagnostic also fails (118/128 pass), so it is not a
candidate production fix. Layer tracing shows identical input embeddings and
first-layer normalization. The first-layer QKV product differs by 0.0050%
relative RMS, while the attention output projection differs by 0.7651%.
`qwen25-small-reference-layer-trace-a2.json` and
`qwen25-small-reference-parts-a2.json` retain the intermediate measurements.
These are localization results, not evidence that a single cause is resolved.

Further inspection found an oracle defect: `.to('cuda', dtype=float16)` cast
HF's deliberately FP32 rotary-frequency buffer along with its parameters.
The corrected verifier moves already-FP16 parameters to CUDA without a dtype
cast and asserts that rotary frequencies remain FP32. All earlier model
comparisons above used the flawed oracle and remain diagnostic history only.
The corrected normal-runtime run, `qwen25-small-reference-rotary-fixed-a2.json`,
still fails: 118/128 checks pass, worst relative RMS 3.3295%, maximum absolute
error 0.85157. Fixing the oracle did not qualify the runtime. Original-shape
and rounded-attention controls are being repeated against the corrected oracle.

The corrected separate-products/rounded-attention diagnostic passes all 128
checks (`qwen25-small-reference-separate-fixed-a2.json`). Keeping the packed
products while rounding attention passes only 120/128. Neither diagnostic is
installed in the runtime.

`qwen25-small-reference-sdpa-a2.json` compares the unchanged packed runtime to
independent HF SDPA execution, retaining the same upstream coefficient decoder,
FP16 parameters, FP32 rotary buffer, prompts and numerical thresholds. All
128 checks pass, with identical greedy tokens, worst relative RMS 0.8475% and
maximum absolute error 0.18360. This qualifies the explicitly named fast-attention
reference scope; it does not supersede the corrected eager-reference failures.
HF and dllm prefill both use PyTorch SDPA, while dllm's decode attention and
model executor remain separate implementations.

`qwen25-small-layout-a2.json` passes all 128 fixed/paged, eager/graph, batch
1/2/4 and chunked-prefill comparisons through context 2048. Worst relative RMS
is 0.5090%, with exact greedy choices. The legacy-named GGUF layout harness has
an explicit `--awq-qualification` mode for this isolated source format.
Shared model-factory tests cover dispatch, packed storage/admission and rejection
of conversion, GEMV, non-four-bit and GPTQ declarations. All 339 regression
tests pass after integration.

`qwen25-small-lifecycle-final-a2.json` exercises the normal model factory and
HTTP server. Three graph/batch/generation/unload cycles release every tracked
GPU weight, scale and zero-point tensor. Allocation falls from 588541952 bytes
to a stable 25559040-byte process workspace. The earlier lifecycle report kept
one 512-byte token allocation in the verifier; the final script releases it.
HTTP registration, exact direct-executor text/token accounting, streaming,
two concurrent requests, cancellation recovery, host restoration, host/GPU
unload and disk reload all pass. Switching uses two aliases of this same
checkpoint; it is not a different-model hot-swap speed measurement.

## Initial matched engine measurement

`compare-*-awq-qwen25-small-first-*` retains the first sequential dllm/vLLM
run: identical pinned AWQ weights, 129 explicit input tokens, 64 output tokens,
context 2048, FP16 execution/KV, concurrency 1/2/4, one warmup and five measured
rounds. Both engines return identical text and token counts across these runs.
vLLM 0.30.0 selects Marlin AWQ and FlashAttention 2. Prefix caching is disabled.
These are initial one-order measurements, not repeated interleaved trials.

| Concurrency | dllm total median (s) | vLLM total median (s) | dllm first output (s) | vLLM first output (s) | dllm aggregate tokens/s | vLLM aggregate tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4992 | 0.2777 | 0.0353 | 0.0352 | 128.1 | 230.2 |
| 2 | 0.5484 | 0.3022 | 0.0437 | 0.0566 | 231.7 | 422.5 |
| 4 | 0.6102 | 0.3130 | 0.0938 | 0.0614 | 416.6 | 812.4 |

The standalone decode profile (`qwen25-small-decode-profile-a2.json`) measures
7.065 ms of GPU kernel time per step: packed AWQ GEMM consumes 5.037 ms and
the dense output head 1.626 ms. Profiling overhead and its shorter prompt make
these diagnostic timings distinct from HTTP measurements. Split-K candidates
are isolated under `scripts/experiments/awq_splitk.py`; normal dispatch is
unchanged until separate matrix, model, layout and serving gates pass.

The first split-K matrix sweep passes 360 checks, but improves down-projections
only about 19% and regresses QKV/gate-up. A separate direct small-row reduction
passes 288 checks. Its 128-output tile reduces single-row gate/up from about
108 to 60 microseconds and down from about 71.5 to 30.7 microseconds; four-row
gains are much smaller and QKV regresses. Both reports are retained under
`qwen25-small-{splitk,gemv}-matrices-a2.json`; neither is a serving result.
The first wider-MMA-tile trial requested 106496 bytes of shared memory, above
A2's 101376-byte limit. Its failure is retained; reduced pipeline stages are
being measured separately.

The resource-bounded wider-MMA run passes 384 checks, but larger tiles generally
regress; its strongest gate/up gain is only about 2.5%. The final direct-reduction
sweep passes 288 checks including three-row batches. The 256-output direct tile
also regresses. The isolated candidate therefore retains the 128-output direct
tile for QKV rows 1–2, output rows 1–4, gate/up rows 1–3 and down rows 1–4,
only for the measured four projection shapes on A2. Other shapes/rows retain
normal dispatch. `qwen25-small-candidate-reference-a2.json` passes all 128
independent HF SDPA checks at 2048, worst relative RMS 0.8120%, with exact
greedy choices. Runtime layout and serving gates remain separate.
