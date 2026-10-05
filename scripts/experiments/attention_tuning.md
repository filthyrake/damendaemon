# A2 Qwen3 decode attention experiments

The matching Qwen3-4B native AWQ profile spends 2.418 ms per batch-four decode
step in attention, plus 0.260 ms merging context splits. Experiments here use
fixed cache layout, 32 query/8 KV heads, dimension 128, context 2048 and batch
one/four. They do not change runtime registration.

`attention_tiles.py` compares independent FP32 math SDPA with existing split
kernels at blocks 64/128/256. It checks positions 0/129/255/256/511/2047,
reordered slots and NaN-filled future cache. Each variant is timed twice in
forward/reverse order using CUDA graphs. The initial 128-token split improves
many cases but retains repeated per-head KV reads.

`--stream` adds scalar online-softmax variants from `attention_stream.py`.
All 72 checks pass, but sequential reduction cost outweighs the reuse benefit;
these variants are not promoted.

`--tensorcore` adds grouped tensor-core variants from `attention_tensorcore.py`.
The initial default pipeline exceeds A2 shared memory (139264 required versus
101376 available); the retained failure precedes the explicit one-stage launch.
The bounded launch passes all 60 matrix checks. At batch four/position 129,
block 128 takes 20.42 microseconds versus 71.34 for the existing split-256 path;
at position 2047 it takes 215.06 versus 583.43 microseconds. Block 64 is better
at some short prefixes (17.54 microseconds at position 129). These are isolated
kernel measurements, not serving gains.

Softmax state and accumulators remain FP32, but probability products are rounded
to FP16 for tensor cores. Complete-model independent oracle and layout checks
must validate that rounding before any promotion. The isolated runner
`run_attention_tensorcore.py` installs shape-limited fixed-cache dispatch around
an existing validation script and writes separate source/selection metadata.
Unsupported features retain ordinary attention; selection counts distinguish
actual experiment coverage from fallback execution.

Reports, including unsuccessful candidates, live in `docs/evidence/awq`.
GitHub issue #49 tracks the Qwen3 performance work.

The initial single-FP16 probability candidate passes 128/128 independent oracle
checks (worst RMS 0.6255%), with 2268 recorded experiment calls and no fallback.
However, it passes only 124/128 cross-layout comparisons: batch-four step two
has 1.0765% RMS in fixed layout and 1.1167% against ordinary paged execution,
repeated in eager/graph modes. All greedy tokens match. The 1% RMS gate stays
unchanged and this candidate is not promoted. Its reports retain the original
source hash. A two-piece FP16 representation of FP32 probabilities is the next
precision experiment; its extra tensor-core product must also be timed.

The two-piece probability candidate also fails four of 128 layout comparisons:
fixed batch-four step two reaches 1.3660% RMS; ordinary paged execution versus
the candidate single-sequence baseline reaches 1.1820%. Greedy tokens still
match. More precise probability products did not solve the full-model layout
sensitivity. Neither grouped candidate is promoted, and no HTTP throughput
claim is made for either. The existing split attention algorithm at block 128
is being checked separately against the same unchanged gate.

Split-128 also fails four layout checks at batch-four step two (maximum RMS
1.2965%), despite matching greedy choices. It is not promoted. The repeated
sensitivity motivates a grouped split-256 candidate that retains the existing
softmax partition and checks bitwise equality to split-256 in addition to the
independent FP32 reference. This is a distinct experiment, not a relaxed gate.

Grouped split-256 and thread-count-only variants pass their independent
numerical checks but fail to improve timing and are not consistently bitwise
identical to the current kernel. At batch four/position 2047, grouped two/four
heads take about 4.686/11.336 ms versus 0.584 ms for the existing path. At
position 256, eight/sixteen warps take 0.144/0.166 ms versus 0.125 ms. These
variants remain isolated experiments. No attention change from this tuning
series is promoted to the qualified runtime.
