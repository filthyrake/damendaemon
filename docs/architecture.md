# Execution and model residency

## Product contract

dllm owns model discovery, lifecycle, scheduling, and the user interface. A
single persistent API accepts a model name on each request. Loading another
model does not require restarting that API. Existing streams drain before their
model is evicted; a deadline or explicit cancellation may terminate them.

GPU execution must be measured on the target hardware. The selected direction
is NVIDIA first and an independent runtime: no vLLM runtime dependency. Reuse
mature kernel libraries where licensing permits instead of rebuilding every
matrix multiplication. The initial test device is the A2 on the user's lab VM;
the A30 is serving other work and must not be used for dllm tests.

## Split responsibilities

- **Control service:** catalog, downloads, checksums, aliases, admission, memory
  budgets, lifecycle, and API/CLI. Never runs a per-token cross-process RPC.
- **Device workers:** persistent device contexts, weight loading, KV allocation,
  GPU execution, sampling, and batched delivery of output. A worker can replace
  a model executor while keeping device initialization alive where supported.
- **Execution plans:** keyed by device architecture, model architecture and
  revision, precision/quantization, parallelism, kernel version, and shape
  buckets. Separate compilation artifacts from live captured GPU graphs.

Language choice follows a profiling-backed prototype. A Python benchmark does
not imply a Python token scheduler. The intended native scheduler/worker boundary
is C++ with CUDA initially; Metal and ROCm require their
own optimized paths. A lowest-common-denominator GPU interface must not constrain
the fast path. First architecture/model coverage should be deliberately narrow.

## Residency and switching

Model state: absent -> disk cached -> host cached -> loading -> ready -> draining
-> host cached or disk cached. Loading/unloading failures must be observable,
release reservations, and leave the service available. Coalesce concurrent loads
of the same immutable model revision. Never publish readiness before weights,
execution plans, and required allocations are usable.

Account for weights, KV cache, scratch space, graph pools, transfer staging, and
a measured safety reserve. Reserve before starting a load. Evict only unpinned,
idle executors; do not free allocations while streams or DMA operations still
reference them. Budget host memory and pinned staging independently. Request
queues must be bounded with deadlines and fair admission across models so model
churn cannot starve a less popular model.

Three different costs need separate measurement:

1. GPU-resident model selection: routing/scheduling; no weight copy required.
2. Host-cached restoration: host-to-device transfer, allocations, and warmup.
3. Disk-cached loading: storage reads, conversion/repacking, transfer, and warmup.

For W bytes of weights and effective transfer bandwidth B bytes/second, W/B is
a transfer lower bound, not a latency promise. Disk/network cold loads cannot
be made instantaneous by a nicer API. Keep frequently used weights resident
when capacity permits; cache device-ready packed weights in host memory when
it does not compromise the machine. Overlap transfers with useful computation
only when benchmarks show a net benefit under bandwidth contention.

KV caches are isolated by model revision, tokenizer, adapter, position settings,
precision, and compatible execution configuration. Reusing an executor's memory
does not make its old KV cache valid for another model. GPU graph reuse requires
compatible shapes and stable valid addresses; caches must be invalidated when
those conditions change. Start with draining active streams, not transparent
mid-generation migration.

## Performance mechanisms to evaluate

Continuous batching, paged KV allocation, chunked prefill with decode latency
budgets, prefix caching, fused kernels, optimized attention, graph capture, and
quantized matrix operations are candidates for the fast path. Speculative decode
is conditional on measured acceptance and total resource cost. None is a claim
that every model or GPU benefits from every mechanism.

Model format is independent of the CLI. Support for safetensors or GGUF must
include the actual architecture, quantization layout, tokenizer, and template;
recognizing the container alone is not model support. Preserve upstream licenses
and attribution for any reused implementation.

## Evidence required before calling it fast

Compare dllm with vLLM and llama.cpp on the same device, model revision,
quantization, prompt/output tokens, context size, and sampling settings. Keep
single-user decode and concurrent throughput as separate results. Record p50/p95
first-output latency, total latency, total output tokens/wall time, failures,
peak GPU/host memory, and warm/cold A -> B -> A switch latency. Token-to-token
latency requires token-level server instrumentation; network chunk arrival is
not a substitute. Prefix reuse, warmup, compilation, and early EOS must be
controlled or reported. Include quality/correctness checks before accepting a
speedup from lower precision or changed sampling.

The first hardware proof should load model A, stream a correct response, switch
to B, switch back to A, and compare both resident execution and transitions with
the baselines. Then test overlapping requests, cancellation, insufficient
memory, and load failures. CPU unit tests cannot satisfy that proof.

## Reference inspection

Inspected local revisions:

- vLLM `84bcbc62644356270aaaa5e2d0237d03adc9bb3a`: README and
  `docs/features/sleep_mode.md`. Sleep can release KV/weights and back up weights
  to CPU; this is useful prior art, not a complete arbitrary-model switcher.
- llama.cpp `836d57176dc699a726c55418e4f96b8ca628e1bf`:
  `tools/server/README.md`. Router mode dynamically selects/loads model instances
  and supports explicit model lifecycle operations.

Primary references: [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
and [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
