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
