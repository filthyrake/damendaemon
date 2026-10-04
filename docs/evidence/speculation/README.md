# Full-target n-gram speculative decoding

Issue #37. Independent implementation inspired by NInfer's separation of draft
proposals from full-target verification. No upstream runtime/source dependency,
trained draft model or vocabulary shortlist is introduced. See the
[prototype](prototype/) for preliminary measurements and deliberately incorrect
proposal tests; oracle figures there are not actual feature performance.

## Supported opt-in path

`DLLM_SPECULATIVE_TOKENS=2|4|8` enables one bounded CUDA verifier graph per
eligible executor; default0 keeps ordinary decoding. Qualified pinned models:
Qwen2.5-0.5B-Instruct (`7ae557604adf67be50417f59c2c2f167def9a775`) and
Qwen3-0.6B (`c1899de289a04d12100db370d81485cdf75e47ca`) on A2. Qwen2 uses its
existing FP16 policy; Qwen3 keeps its qualified FP32 arithmetic policy.

The HTTP scheduler uses speculation only for one active greedy request with
unquantized dense contiguous KV. It falls back for stochastic sampling,
penalties/biases, structured output, tools/tool history, text stops, logprobs,
multiple active requests, paged KV, chunked/prefix scheduling, unsupported
architectures/devices, disabled graphs and insufficient context/stream space.
Metal, paged, quantized and graph-disabled executors allocate no verifier. The
chunked/prefix scheduler currently leaves an unused verifier resident when the
option is enabled; this is bounded overhead, not speculative prefix support.
The serialized non-batched Runtime generation API is unchanged.

Proposals search up to1024 recent prompt/output tokens for a1–4 token matching
suffix and copy its following tokens. They are only guesses: each block runs
the complete target vocabulary at distinct causal positions in the same KV
slot. Matching drafts and the target's first correction/bonus token are emitted
through the ordinary per-token stop/usage/stream path. Cache length commits only
the consumed prefix; rejected/filler/post-EOS KV remains causally invisible and
is overwritten when needed. This is greedy verification, not a proof of
stochastic acceptance or recurrent-state rollback.

Proposal count adapts to free capacity in a bounded stream queue. Eligible
requests can hold two verification blocks (at most16 events); ordinary requests
retain eight. Two consecutive blocks without a matching
draft trigger eight ordinary steps before trying again. Response timing includes
`speculative_blocks`, `speculative_proposed_drafts` and
`speculative_matched_drafts`. Matches count verification matches, which can
include matches after an EOS encountered while emitting the block; they are
not billed/generated token counts. Usage continues to count emitted tokens.

## Initial complete HTTP performance (eight-event buffer)

A2, same pinned checkpoint per pair,512 context,64 output tokens, one client,
resident model, six measured alternating baseline/optimized rounds after warmup.
Baseline temporarily disables the verifier on the same executor between
completed requests. Both modes include tokenization, prefill, proposal work,
GPU execution, stream-buffer scheduling and HTTP serialization. Qwen3 thinking
is disabled in both. Full response text, finish reason and usage match exactly.

| Model/workload | Ordinary ms | Speculative ms | Ordinary output tokens/s | Speculative output tokens/s |
|---|---:|---:|---:|---:|
| Qwen2 repeated sentence | 466.085 | 156.577 | 137.31 | 408.75 |
| Qwen2 explanation | 465.730 | 456.442 | 137.42 | 140.22 |
| Qwen2 code/list | 465.326 | 324.732 | 137.54 | 197.09 |
| Qwen3 repeated sentence | 575.197 | 290.418 | 111.27 | 220.37 |
| Qwen3 explanation | 572.796 | 573.591 | 111.73 | 111.58 |
| Qwen3 code/list | 576.554 | 450.731 | 111.00 | 141.99 |

Throughput here is output count divided by median complete request latency,
including prefill. Repetitive prompts intentionally expose n-gram reuse; this
is not a broad workload benchmark or a new comparison with vLLM/llama.cpp.
Ordinary prose is approximately unchanged in this small sample and can regress
when proposals are poor. This remains opt-in. The initial strict eight-free-
entries gate suppressed most verification and only improved repetition about9%;
that failed scheduling experiment is preserved in `serving/`.

## Correctness, bounds and lifecycle

The prototype covers all2/4/8 widths, all-accepted, real n-gram and forced-reject
blocks, EOS, budget and context tails. The production verifier separately passes
18 cases per model on nonzero slots1/3, including alternating acceptance and
rejection; another slot's entire KV stays unchanged. All generated tokens equal
ordinary greedy decoding in these tests.

Live HTTP tests pass exact response/stream equivalence, seeded sampling and
other fallback behavior, slow-consumer backpressure, disconnect recovery,
four-request concurrency, graph reclamation on unload and host restoration.
Paged/chunked/prefix/Q8/graph-disabled fallback tests pass on A2; Metal fallback
passes on M4. Interleaved narrow-head candidate scoring leaves ordinary generation
unchanged and cannot alter the captured full-vocabulary verifier.

Verifier preparation adds roughly0.16–0.18 seconds after an already-resident
executor and retains about8.13MiB allocated device memory in the measured slot
probe. This includes warmup and graph capture; model load is separate. Weak-
reference checks prove verifier/graph objects are released on destruction and
real model unload. Cold activation is therefore a tradeoff, not a free speedup.
Full unit/HTTP regression suite:214 tests pass.

Reproduction scripts: `verify_speculative_service.py`,
`verify_speculative_slots.py`, `verify_speculative_fallbacks.py` and the retained
prototype. Raw trials, exact outputs, fallback modes and graph memory evidence
are in `serving/`. No multi-GPU, recurrent, stochastic or quantized speculation
claim is made by these results.

## Two-block buffering supersedes the first serving measurements

The initial eight-event queue forced extra single-token steps while a consumer
was still draining the previous verified block. Eligible requests now retain
room for two blocks: at most16 events for width8, while ordinary/ineligible
requests retain eight. The existing queue object is resized under its lock
before the first event, avoiding a consumer switching between queue objects.
Proposal count remains bounded by currently free capacity; no unbounded buffer
or forced client-speed assumption is introduced.

A related pre-existing completion race is fixed: a producer can enqueue its
last block and finish after a consumer's empty observation but before its done
check. The consumer now drains any queued events before yielding completion.
A deterministic unit test injects exactly this interleaving and proves the last
token is delivered before the final event.

The same six-round alternating A/B HTTP suite passes again. New measurements:

| Model/workload | Ordinary ms | Speculative ms | Ordinary output tokens/s | Speculative output tokens/s |
|---|---:|---:|---:|---:|
| Qwen2 repeated sentence | 462.351 | 104.388 | 138.42 | 613.10 |
| Qwen2 explanation | 465.309 | 448.621 | 137.54 | 142.66 |
| Qwen2 code/list | 466.083 | 303.487 | 137.31 | 210.88 |
| Qwen3 repeated sentence | 565.613 | 242.901 | 113.15 | 263.48 |
| Qwen3 explanation | 564.664 | 566.247 | 113.34 | 113.02 |
| Qwen3 code/list | 566.080 | 423.617 | 113.06 | 151.08 |

This yields4.43× /1.54× on the Qwen2 repetition/code cases and2.33× /1.34×
on Qwen3. All prior workload/precision limits still apply; prose remains roughly
unchanged. The earlier eight-event results stay preserved as historical evidence.

`verify_speculative_backpressure.py` deliberately reaches all16 occupied events
on each GPU-qualified model, proves a second request finishes while the first
consumer is blocked, then checks exact token draining. Closing a full stream
finishes cancellation and a subsequent request produces the reference output.
The repeated full HTTP suite also passes streaming, concurrency, fallback and
model lifecycle checks. Updated regression suite:215 tests pass. Raw reports
are the `sixteen-events` and `backpressure` files under `serving/`.

## Larger dense Qwen3 qualification

Qwen3-1.7B (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`) passes the same
18 verifier width/slot/acceptance cases and complete HTTP suite on A2, using its
existing qualified FP32 arithmetic policy. The six-round alternating A/B test
keeps 512 context, 64 output tokens, thinking disabled and a resident model.

| Workload | Ordinary ms | Speculative ms | Speedup |
|---|---:|---:|---:|
| Repeated sentence | 1441.382 | 310.523 | 4.64× |
| Explanation | 1432.256 | 1426.299 | 1.00× |
| Code/list | 1449.557 | 1087.017 | 1.33× |

Response text, finish reason and usage match exactly. Streaming, fallback,
concurrency, disconnect recovery, graph reclamation and host restoration also
pass. These are the same narrow workloads, not a general speedup or cross-engine
comparison. Raw reports are the `qwen3-1.7b` files under `serving/`.
