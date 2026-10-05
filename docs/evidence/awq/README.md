# AWQ checkpoint format groundwork

GitHub issue #49 tracks import and execution qualification. The format module
is not yet connected to model registration or inference; AWQ runtime support
and performance parity are not claimed by these host checks.

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
