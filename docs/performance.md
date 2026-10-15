# Performance tuning

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Measure first-token latency and decode rate separately. Optional caches and scheduling policies trade memory, latency, and throughput; use the linked evidence to choose settings.

## Chunked prefill and prefix reuse

Opt-in scheduler chunking is available with `dllm serve --prefill-chunk-size 256`.
On supported dense rotary decoders, the worker processes at most that many
prompt tokens before returning to active decoding, and samples the first output
only after the full prompt is cached. The default is 0 (full prefill). This can
reduce stream stalls when a long prompt joins, at a cost to long-request latency.
It preserves FIFO model/management barriers and cancellation/slot reuse. Other
architectures, including Phi-3 pending a numerical fix, retain full prefill.
This does not implement paged KV or prefix caching. Measurements and exact
qualified model/precision combinations are in the architecture evidence.

For repeated prompt prefixes, `dllm serve --prefix-cache-tokens 4096` enables
bounded GPU KV snapshots on the same decoders qualified for chunked prefill.
The budget is per resident model, with at most four entries. Exact repeats can
reuse final prompt logits; divergent suffixes resume from matching cached KV.
The cache copies KV into independent request slots, so decoding one request
cannot modify another's snapshot. Evicting a model also evicts its prefix cache.
`dllm_timing.cached_prompt_tokens` reports reuse; ordinary usage still counts
all input tokens. The default is disabled. This uses additional GPU memory and
copies; it is not paged KV or zero-copy sharing. Under memory pressure, caching
is skipped while ordinary inference continues. Numerical qualification and
latency measurements are in the architecture evidence.

OLMoE-1B-7B-0924-Instruct Q8 on A2 now supports these opt-in chunking and prefix
features at context up to 2048 and batch capacity up to four. Short expert
prefill and packed concurrent prompts are selected automatically within their
measured 128-token bounds. Prompts or uncached suffixes of at most four total
tokens reuse routed projections, avoiding sparsely populated grouped tiles.
Repeated default-scheduler HTTP tests show a further
17% C4 total-latency improvement from packing; single-request latency is
unchanged by packing. A separate warm prefix-hit test reduces server first-token
time for a repeated 1536-token prompt from 1792 to 6 ms. These are different
workloads; see [OLMoE scope, measurements and independent proof](architecture-parity.md#olmoe-checkpoint-qualification).
On the measured OLMoE mixed-request workload, 128-token chunks reduce the maximum
streaming pause from 1.80 to 0.32 seconds but increase both requests' completion
times. Full prefill remains the default.

## Shared paged KV cache (CUDA, opt-in)

`dllm serve --kv-cache-tokens 2048 --context 2048 --batch-size 4 ...` allocates
one shared 2048-token KV pool per resident model, instead of four separate
2048-token caches. The budget must be a multiple of 16 and fit at least one
page per slot. `DLLM_KV_CACHE_TOKENS=2048` selects the same policy; an explicit
`--kv-cache-tokens 0` restores fixed slots. Fixed slots remain the default.

The CUDA dense path supports Qwen2, Qwen3, Llama and Mistral, including supported
weight quantization policies. Qwen2, accurate Qwen3, Llama and Mistral have
independent paged reference proof. Page reservations cover each request's prompt plus maximum
output, rounded to pages. Requests wait in order when other requests temporarily
own the capacity; requests larger than the entire pool are rejected. Completion,
cancellation, failures and model eviction release ownership. CUDA decode graphs
read the live page table without rebuilding graphs at page boundaries.

Add `--prefill-chunk-size 128` to interleave bounded prompt chunks with active
decodes while keeping each request's page reservation. Qwen2, accurate Qwen3
and Mistral Q4 pass chunked reference checks. CUDA Llama defaults to
`DLLM_LLAMA_PRECISION=mixed`: FP32 residual accumulation and precise elementwise
arithmetic resolve the standard FP16 path's numerical gate failure while
retaining FP16 weights, Tensor Core projections and FP16 KV. The mixed policy
passes Q4 full/chunked fixed and paged replay at context 2048, FP16 full/chunked
replay at context 8192, and Q4 prefix reuse. `DLLM_LLAMA_PRECISION=standard`
restores the legacy arithmetic, including its known chunked numerical failure.
Add `--prefix-cache-tokens 4096` for bounded copied prefix snapshots alongside
the page pool. Snapshot storage has its own budget and consumes additional GPU
memory; requests still reserve their full prompt/output capacity before reuse.
Other architectures and Apple Silicon are not supported by this paged path.
It does not quantize KV or share pages between different prompts. Full prefill
uses fused current-token K/V outputs; offset chunks use a native bulk page gather, and
decode reads pages directly. Reducing
the pool saves allocated GPU memory but may reduce simultaneous admission for
long requests. See [paged-cache evidence](evidence/paged/README.md).

## Optional n-gram speculative decoding

`DLLM_SPECULATIVE_TOKENS=8` enables bounded full-target CUDA verification for
qualified unquantized Qwen2/Qwen3 greedy requests; 2/4 are also supported and 0
(the default) disables it. It reuses repeated prompt/output fragments as draft
proposals and verifies every accepted token with the full target vocabulary.
Llama3.2-1B additionally requires explicit `DLLM_LLAMA_PRECISION=float32`;
its default mixed precision fails the exact-token gate and remains excluded.
The higher-precision policy retains FP16 stored matrices but uses FP32 KV and
residual arithmetic. It adds cost on ordinary prose; see the measured
[precision tradeoff and numerical evidence](evidence/speculation/llama/README.md).

The serving path falls back for multiple active requests, sampling, constraints,
tools, penalties, logprobs, text stops and unsupported memory/device paths.
Cold graph preparation costs about 0.17 seconds and 8.13 MiB in the A2 probe.
Measured complete HTTP gains reach roughly 4.4× on the tested Qwen2 repetition
case and 1.5× on its code case; ordinary prose is approximately unchanged and
poor proposals can regress. [Exact-output, fallback, lifecycle and performance
evidence](evidence/speculation/README.md).

## Higher-precision prompt processing

Long CUDA FP32 prompts use fused grouped-query attention to avoid quadratic
temporary storage. Full 8K Llama prefill now fits the A2; at 2K, measured complete
HTTP latency drops 23% at the same precision.
[Reference, memory and request-level evidence](evidence/fp32-attention/README.md).

The higher-precision dense path also uses compensated Tensor Core projections
for longer prompts. Measured complete HTTP latency drops a further 19–31% on
the tested Qwen3/Llama cases, with independent reference and exact-output gates.
[Projection accuracy, memory and full-request measurements](evidence/precise-prefill/README.md).

## Optional prompt coalescing

`dllm serve --batch-wait-ms 1` waits up to 1 ms before an idle model group starts,
allowing closely arriving requests to share prompt processing. The default is
0 (disabled), and the supported range is 0–10 ms. It does not wait between active
decode steps or cross queued model/management barriers.

Normal A2 server comparisons reduce two-request latency by 14% for TinyLlama
and 18% for Mistral, with four-request total latency roughly tied. Observed
single-request first-output differences are about 1 ms for TinyLlama and
1–7 ms for Mistral; coalescing can also delay the earliest concurrent request's
first output while bringing later requests forward. See
[the measured tradeoffs](evidence/gguf/README.md).


## Prompt graph cache

For repeated short Qwen2/Qwen3 prompts on NVIDIA, opt-in
`DLLM_PREFILL_GRAPHS=4` caches up to four prompt-length/slot CUDA graphs per
resident executor. `DLLM_PREFILL_GRAPH_MAX_TOKENS` defaults to128. First captures
are expensive; dynamic workloads may not amortize them. Quantization, paged KV,
and packed/chunked prefill retain their existing paths. Direct token scoring
uses separate graph keys for each ordered candidate set, sharing the same cache budget.
[Correctness, memory release and matched HTTP performance](evidence/prefill-graphs/README.md).

## Direct generation graphs

Direct CUDA generation (`dllm run` and `Runtime.generate`) reuses the native
one-request decode graph. A2 measurements show 2.5–3.0× faster Qwen3-0.6B FP32
decode and about 2% faster GPT-OSS decode versus the previous scalar dispatch;
HTTP batching already used these graphs. [Scope and evidence](evidence/scalar-decode/README.md).

## Fixed-rate HTTP load and observed stream stalls

`dllm bench` defaults to closed-loop load: each completed request frees a client
slot for the next one. Add `--arrival-rate` to offer requests on a fixed schedule
independent of response completion:

```sh
dllm bench --base-url http://127.0.0.1:8088/v1 --models chat \
  --requests 100 --concurrency 8 --arrival-rate 2 --max-tokens 128
```

This is an example workload, not a qualified load level. `--concurrency` caps
client requests in flight, including those waiting inside the server; it is not
an active-user count. Arrivals finding all client slots occupied become explicit
`client capacity` failures rather than waiting in a hidden client queue. Results
retain every scheduled index, including drops; any failure makes the CLI exit
nonzero while still printing its report. HTTP failures are counted separately.
The report includes scheduled offsets and actual dispatch delay; inspect those
lags before treating the client as a faithful load generator. The declared
arrival rate is not a claim that every request reached the server on time.
Wall time and throughput include draining the last requests.

Schema version2 adds p99 to latency distributions and records intervals between
client-observed SSE output events. Role-only, finish and usage events do not
count as output. Content, reasoning and tool-call events do. A single event may
contain several tokens, so these gaps are **not per-token GPU timings**. Token
throughput still uses reported completion-token usage; missing usage stays
unknown. Successful-request summaries include pooled gap percentiles and the
distribution of each request's largest gap; raw per-request gaps retain stalls
and failed partial streams. Requests with fewer than two output events have no
gap measurement, not an invented zero. Gap collection fails the request after
100,000 output events to bound additional sample memory. These measurements do
not automatically certify a sustained-user SLO or calculate SLO goodput.

To evaluate a declared client-observed profile, supply all three options before
running: `--slo-first-output SECONDS`, `--slo-output-gap SECONDS`, and
`--slo-request-rate TOKENS_PER_SECOND`. They bound first visible output, the
largest SSE output-event gap, and the minimum **whole-request average** output
token rate (usage tokens divided by total duration, including prefill).
Schema version 3 evaluates fixed-rate requests from their **scheduled arrival**:
observed dispatch delay is added to both first-output time and total duration.
For example, 1 second of dispatch delay plus 0.25 seconds to first output is
1.25 seconds against the first-output budget. SSE gaps remain unchanged.
Missing or invalid dispatch timing makes a scheduled request unmeasurable.
Closed-loop profiles retain request-start timing. Each measured profile result
names its `timing_origin` and adjusted first-output/request durations; raw result
fields and ordinary latency summaries remain request-relative. Workload metadata
identifies both origins. Version 2 profiles excluded dispatch delay, so their
fixed-arrival verdicts must not be compared as if they used the same timing policy.

They do not implement the separate post-prefill per-user speed target in #59.

The report retains ordinary throughput and adds `client_profile`: per-request
pass/fail/unmeasurable/error reasons, counts, and qualifying requests/tokens per
wall second. Every scheduled request remains in the outcome counts; client
capacity drops and server failures never qualify. Missing usage, missing first
output, or fewer than two output events makes the request unmeasurable. A
single coalesced output event therefore cannot falsely pass a stall budget.
Exact threshold equality passes. An explicitly requested profile exits nonzero
unless every request passes, including when HTTP succeeded but measurements
were incomplete. Fix the profile before comparing engines; no automatic budget
relaxation occurs. SSE transport behavior still prevents claiming per-token GPU
latency or sustainable-user qualification from this report alone.

When a server supplies `dllm_timing.queue_seconds`, the benchmark retains it as
`reported_server_queue_seconds` and reports its successful-request distribution
and coverage count. This is server-reported worker-queue delay, separate from
client dispatch lag and client-observed first-output latency. Older or other
engines may omit it; absent values remain unknown. Nonfinite, negative, boolean
or malformed reported timings fail validation rather than contaminating the
measurement. Queue delay is not subtracted from latency or used to relax a
client profile.
