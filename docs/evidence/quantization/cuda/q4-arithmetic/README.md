# Q4 arithmetic isolation

Tracked in issue #23. The expanded CUDA FP16 Qwen2.5-0.5B trace has three greedy
failures in both sequential and packed layer-partition execution, despite RMS
and maximum errors staying within the unchanged numerical limits. These
experiments investigate that gap; they do not mark it resolved or change serving.

The pinned model revision is `7ae557604adf67be50417f59c2c2f167def9a775`, with
Q4_0 coefficients, context 512, four slots and A2/A30 stages split 12/12.
`scripts/experiments/diagnose_q4_head.py` replays the same 282 independent
teacher-forced checks eagerly. Each output head variant sees exactly the same
normalized hidden state within an event. Diagnostic dense copies restore the
stored Q4 coefficients to FP16; the FP32 head then promotes those coefficients
and activations, with TF32 disabled. No variant's output feeds back into the
trace, and no gate or independent target is changed.

| Body projections | Native packed head failures | Dense FP16 head failures | Dense FP32 head failures |
|---|---:|---:|---:|
| Native packed | 3 | 3 | 3 |
| Temporary dense FP16 | 1 | 1 | 1 |
| Native packed + unfused normalization | 3 | 3 | 3 |
| Dense FP16 + unfused normalization | 0 | 0 | 1 |
| Contiguous packed body products | 0 | 0 | 2 |

With native body projections, native and dense FP16 heads fail the same events
61, 135 and 159. The FP32 head fixes events 61/135 but introduces failures at
93/144; event 159 remains wrong. Those new cases include reference FP16 ties,
so removing the final rounding point is not a general solution.

With temporary dense body projections, native and dense FP16 heads fail only
event 135, while the FP32 head fails event 93 instead. This implicates body
arithmetic as a contributor, but does not establish the complete cause. A dense
output head alone does not solve the issue, and a larger dense-body fallback
still does not pass on its own. Neither is promoted. These are correctness diagnostics,
not performance measurements, and they intentionally use extra GPU memory.

Reproduce with the independent tiny-prefix Q4 trace and `--output FILE`, then
repeat with `--dense-body`. `--packed` selects packed prompt admission; the
recorded controls here use the sequential default. All jobs are bounded and
run sequentially without altering the existing services.

The existing experimental `q4_prefill.single_dot` kernel changes body projection
accumulation order from separate low/high nibble products to contiguous K-order
products. It keeps coefficients packed, FP32 accumulation and FP16 outputs,
and retains the native packed output head. That candidate passes all 282 eager
checks without replacing normalization. The explicit `run_q4_body.py replay`
dispatch also passes 282 checks with real CUDA graphs and 22 observed packed
stage calls. This is evidence for a candidate, not proof of complete equivalence
on arbitrary prompts or a production promotion. Its serving and performance
qualification are separate from these isolation controls.

The explicit candidate passes 19 ordinary same-host HTTP checks with packed
prefill enabled, including streaming, concurrency, cancellation and model swaps.
Changed-weight qualification passes 410 independent checks across four alternating
weight sessions and a fresh recovery after partial-upload failure. Three actual
rebinds preserve captured graph storage; the partial executor is unpublished.
These lifecycle timings are not a matched hot-swap benchmark.

The same contiguous-body candidate passes 282 checks on the single A2 with
graphs as well. However, its matched A2/A30 HTTP measurement rejects it for
production on performance grounds. Both modes keep prompt packing disabled,
capacity four, context 512, 16 input IDs and 32 output tokens; paired/contiguous/
contiguous/paired launches provide ten measured rounds per mode/concurrency.
Text and usage match. Startup/capture and warmups are excluded.

| Concurrency | Body kernel | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Current paired | 28.17 | 162.48 | 196.35 |
| 1 | Contiguous candidate | 33.41 | 859.39 | 37.21 |
| 2 | Current paired | 43.93 | 201.69 | 312.58 |
| 2 | Contiguous candidate | 56.86 | 912.50 | 68.90 |
| 4 | Current paired | 107.63 | 289.04 | 438.81 |
| 4 | Contiguous candidate | 133.22 | 988.18 | 128.48 |

The first candidate's large throughput regression is unacceptable despite its
passing correctness gates. It remains an explicit experiment; production Q4
dispatch and conservative prompt-packing default are unchanged. Reproduce with
`bench_partition_prefill_service.py --compare q4-body --quantization q4_0`.

## Packed-load revision

`single_dot_paired_load` loads each packed byte once and each block scale once,
then reconstructs contiguous K-order values for the dot product. It passes all
282 graph-backed model checks and all 20 synthetic body-matrix checks. The
synthetic checks require exact FP16 output equality with the first contiguous
candidate and retain the independent dense RMS gate.

Warm CUDA-graph profiling on the A2 exposes the remaining cost. For one row:

| Projection (outputs × inputs) | Current paired, µs | First contiguous, µs | Packed-load revision, µs |
|---|---:|---:|---:|
| QKV (1152 × 896) | 8.88 | 79.92 | 19.39 |
| Attention output (896 × 896) | 8.78 | 79.71 | 19.34 |
| Gate/up (9728 × 896) | 59.33 | 629.02 | 145.78 |
| Down (896 × 4864) | 44.81 | 460.32 | 103.18 |

The revision is substantially faster than the first candidate but still
2.2–2.5 times slower than current body kernels on these small-row cases.
Compiler metadata reports 211 registers versus 128 for current kernels, with
zero spills in both. A launch/tile sweep is needed before considering another
serving comparison. These synthetic matrix measurements are not model
throughput claims. `benchmark_q4_kernels.py --body-candidates` reproduces them.

The bounded tile/launch sweep (`--tune-body`) keeps K tiles at 128 and checks
exact contiguous-output equality for every configuration. On the two expensive
one-row projections, its best results are gate/up 131.32 µs versus current
59.27 µs, and down 83.19 µs versus 45.13 µs. Reducing registers or changing
occupancy alone does not remove the regression. No tested setting is promoted.

## Adjacent-nibble layout experiment

A diagnostic byte permutation packs consecutive K positions together instead
of Q4_0's separated half-block nibbles. It preserves integer values, block
scales and packed byte count. The corresponding kernel avoids the per-tile
half-block transpose while preserving contiguous accumulation order.

The initial A2 profile checks two body projection shapes at one/four rows and
three layout-specific launch configurations. Every candidate exactly matches
the original contiguous output. Best one-row timings are gate/up 65.21 µs
versus current 59.13 µs, and down 48.65 µs versus 44.53 µs. At four rows these
are 66.45 versus 60.43 µs and 49.94 versus 44.83 µs. The remaining gap is about
9–11%, a substantial improvement over the earlier candidates but still a
regression. These measurements exclude repacking cost.

The initial measurements above apply only to the matrix profiler and exclude
activation, serving and hot swaps. No checkpoint format, production weight
layout or default has changed.

A subsequent bounded sweep found an exact-output gate/up tile slightly faster
than current kernels (58.05 versus 59.15 µs for one row, 59.43 versus 60.68 µs
for four). The best down tile remains slower (48.52 versus 44.59 µs for one
row). Every timing candidate first passes coefficient-error and exact-output
checks; failed configurations are retained as failures and not timed.

The explicit `adjacent-gate` model experiment changes only gate/up projections;
it still fails three of 282 checks, at events 93, 135 and 159, and is rejected.
`adjacent-all` changes all four body projection types and passes all 282 checks
with graphs and 22 actual packed stage calls. It retains the current output
head. The prototype keeps canonical Q4_0 weights plus an additional packed
cache, so its memory use is higher; it is not a production storage format.
Its explicit rebind hook refreshes stable cached buffers or closes the executor
on failure.

That prototype passes 19 real HTTP checks and 492 independent lifecycle checks:
four alternating weight sessions, fresh recovery after partial canonical upload,
and fresh recovery after a failure on the second packed-cache refresh. Both
partially modified executors are unpublished; captured packed-buffer pointers
remain stable across successful rebinds. Host residency releases completely.
The additional packed cache is **178,913,280 bytes across both stages**. That
extra allocation is specific to this experiment and must not be omitted from
any memory claim. Performance qualification remains separate.

The cached prototype's matched HTTP run uses the same Q4 weights, A2/A30 pair,
context 512, four slots, 16 input IDs and 32 outputs, with prompt packing disabled
in both modes. Paired/adjacent/adjacent/paired launches provide ten measured
rounds per mode/concurrency. Generated text and usage match.

| Concurrency | Body kernel | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Current paired | 28.63 | 165.80 | 192.53 |
| 1 | Adjacent cache | 27.12 | 164.18 | 194.39 |
| 2 | Current paired | 47.66 | 203.05 | 310.38 |
| 2 | Adjacent cache | 52.45 | 206.64 | 303.42 |
| 4 | Current paired | 107.72 | 289.13 | 437.49 |
| 4 | Adjacent cache | 105.17 | 278.98 | 453.68 |

The original large slowdown is gone. These small, mixed differences are treated
cautiously: throughput is about 1.0% higher at concurrency one, 2.2% lower at
two, and 3.7% higher at four. This is not evidence of a universal speedup, and
the extra 171 MiB packed cache remains an important cost of this prototype.

## Existing-buffer prototype

The explicit `adjacent-inplace` variant repacks the 96 existing body-weight
buffers once during activation and again after canonical weight uploads. It
retains zero additional packed-cache bytes. Large prefills expand the adjacent
representation temporarily for the existing dense fallback. Host checkpoint
bytes and their cache format remain canonical Q4_0; device pointers remain
stable for captured graphs. Repacking still needs temporary workspace.

This variant passes 282 graph/packed-prefill checks, 19 live HTTP checks and
492 lifecycle checks, including both partial canonical upload and partial
in-place repacking failures. Failed executors are unpublished, and fresh
recovery matches independent reference outputs. The prototype remains separate
from production dispatch while its complete policy is measured.

The complete existing-buffer policy comparison enables adjacent body products
and packed prompt admission together. The control retains paired body products
and sequential prompt admission. Same weights, A2/A30 pair, context 512,
capacity four, 16 input IDs and 32 output tokens; ABBA launches provide ten
measured rounds per mode/concurrency. Text and usage match.

| Concurrency | Policy | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Existing | 28.34 | 165.71 | 192.55 |
| 1 | Adjacent in-place + packed | 27.62 | 167.09 | 191.01 |
| 2 | Existing | 42.72 | 197.06 | 319.31 |
| 2 | Adjacent in-place + packed | 45.03 | 207.85 | 304.71 |
| 4 | Existing | 104.56 | 282.40 | 447.80 |
| 4 | Adjacent in-place + packed | 63.84 | 239.60 | 527.13 |

At concurrency four, throughput improves 17.7%, first text 39.0% and request
latency 15.2%. Single-request throughput is roughly unchanged (-0.8%); at two
requests it regresses 4.6% and request latency increases 5.5%. This is a combined
arithmetic/admission policy comparison, not a kernel-only speedup. The runtime
integration must independently pass correctness, lifecycle and serving gates.

## Integrated stage implementation

`DLLM_Q4_STAGE_ADJACENT=1` selects the runtime implementation in
`dllm/q4_stage.py` and `dllm/q4_stage_kernels.py`. It is restricted to the qualified
Qwen2.5-0.5B stage geometry, split 12/12, context at most 512 and up to four
slots. Canonical host/checkpoint data stays unchanged; only device body weights
are repacked. The head retains its current path. Rebind refresh now occurs
inside the existing failure boundary before successful metadata publication.
Distributed startup includes the actual layout in its worker-agreement digest.

The integrated implementation independently passes 282 expanded CUDA graph
checks with 22 packed stage calls, 19 real HTTP checks, and 492 changed-weight
and failure-recovery checks on A2/A30. All 96 body allocations and captured graph
storage remain stable across successful rebinds. Failed partial uploads and
partial repacks are unpublished; fresh recovery matches independent references.
These lifecycle times include activation/cache work and are not a matched
hot-swap speed comparison. The layout flag and Q4 prompt packing still default
off while the remaining production-path qualification runs.

The integrated matrix dispatch also passes 88 real GPU cases across A2 and A30:
all four body projection shapes at 1, 2, 4, 16, 17, 32, 64, 128, 129, 256 and
512 rows. Worst relative RMS versus independently restored FP16 coefficients
is 0.000419 (limit 0.001). Every large-prefill fallback at 129/256/512 rows is
bit-exact with the dense FP16 control. QKV cases include bias. These synthetic
checks exercise layout and dispatch boundaries; they are not throughput claims.
