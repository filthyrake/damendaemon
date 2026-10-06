# Qwen3.8-27B Q4 prefill diagnosis

Issue [#68](https://github.com/filthyrake/dllm/issues/68) owns current execution
status. These are bounded experiments on HOMER Radeon 8060S/gfx1151, PyTorch
2.11.0+rocm7.13.0a20260425, pinned model
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. No rejected attention candidate is
imported by the runtime. Production admission remains unchanged.

## Full-model attribution

Engine `1234e01`, Q4, context4096, chunk127, no graphs: all84 independent
reference targets pass. The [trace summary](q4-prefill-diagnosis-4k.summary.json)
records53 completed chunks and no unfinished spans. Inclusive instrumented
prefill time is153.61s, projections79.01s (gate/up33.82s, down18.42s), SDPA5.02s.
Parent and child durations overlap. Instrumentation adds synchronization and I/O;
these times must not be compared as normal serving latency. This shows completed
work and dominant operations at4K, not definitive full-model32K attribution.

The [raw trace](q4-prefill-diagnosis-4k.jsonl.gz),
[replay result](q4-prefill-diagnosis-4k.replay.json), and
[guard report](q4-prefill-diagnosis-4k.guard.json) retain details. Peak allocated
memory was19.08GiB. Host available memory stayed above59.2GiB; no watchdog trigger.
Allocator28GiB, process64GiB/no swap, child900s/outer930s. The owned process exited.

## Attention-only controls

Each cell uses24 query heads,4 KV heads,127 queries,256 head dimension, FP32.
Selected rows compare against direct FP32 dot/softmax/value products with
relative RMS and max absolute error both below1e-4. This is a selected-row
primitive check, not full-model qualification. Timings include necessary KV
expansion or views. Each comparison has two warmed paired rounds. The raw
JSONL outputs retain order and all samples. Each job had a2GiB allocator,
8GiB/no-swap process scope,20GiB host watchdog and180s child deadline.

| KV length | Native GQA | Expanded KV | Grouped KV views | Fixed streaming tile |
|---|---:|---:|---:|---:|
|4096|~13ms|~22ms|~27ms|see raw evidence|
|16384|~68ms|~88ms|~109ms|~140ms|
|32768|allocator OOM|~183ms|~218ms|~327ms|

Native samples vary slightly between separate comparisons; raw paired results
are authoritative. Expanded KV passed32K at1.79GiB peak allocated. Grouped views
used297MiB and the compensated streaming tile294MiB, but both were slower.
These are negative latency results: none is promoted. Native32K has no completed
latency measurement under this cap. The historical host OOM cannot be attributed
solely to these attention-only results.

Evidence: [expansion](q4-short-query-diagnosis.guard.stdout),
[grouped views](q4-grouped-query-diagnosis.guard.stdout),
[streaming tile](q4-streaming-query-diagnosis.guard.stdout), with matching guard
JSON files. `diagnose_short_query_attention.py --comparison expanded|grouped|streaming`
replays the retained comparison shapes and order; earlier runs selected their
comparison in the source before the CLI selector was added. The streaming
candidate uses three BF16 products with FP32 softmax/state and therefore changes
arithmetic; the SDPA candidates retain FP32. No numerical threshold was relaxed.

## Accepted primitive, integrated qualification interrupted

The grouped **FP32 matrix** control (distinct from grouped SDPA views) reuses a
KV matrix across flattened query-head rows. At16K, paired medians were67.43ms
native versus46.64ms grouped; the32K grouped cell completed at105.67ms and681MiB
peak. Maximum selected-row relative RMS was2.67e-6. See
[raw control](q4-grouped-matmul-diagnosis.guard.stdout).

Commit `d75100b` integrates this calculation with a256MiB score/probability
scratch bound, splitting query rows while preserving the complete causal prefix.
Dispatch is restricted to ROCm compensated hybrid prefill, more than one query,
and at least1024 cached tokens. Other devices, precision modes, short context
and decode retain their paths. All550 local tests pass. This is still pending
full-model GPU qualification under issue74.

The existing-kernel chunk512 control passed84/84 independent4K targets
(max RMS0.003656, max absolute0.085692), but its116.65s total replay did not
establish an improvement over the retained114.35s chunk127 result. These are
single replay runs, not a competitive benchmark or paired full-model speed claim.
See [chunk512 result](q4-prefill-chunk512-4k.guard.stdout).

During the integrated `d75100b` chunk127 replay, HOMER became unreachable by SSH
and ping from the Mac and by ping/HTTP from labnode3; neighbor discovery failed
and DNS still pointed at the same address. The [incident record](q4-grouped-prefill-4k-unreachable.json)
distinguishes observed reachability from unknown cause. The remote replay result,
watchdog outcome, cleanup and service health could not be retrieved. No32K run
followed. Full-model qualification must not be inferred from primitive or CPU
results. The projection control script was prepared locally but never run.
