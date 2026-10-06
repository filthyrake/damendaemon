# Execution and model residency

## Current execution architecture

dllm owns its forward passes, model lifecycle, scheduling and serving API.
The runtime does not wrap vLLM, llama.cpp or Ollama. Python currently implements
control and scheduling; GPU work runs through Triton, native CUDA/C++ or Metal
kernels and PyTorch GPU operations. PyTorch provides tensor storage, CUDA graph
capture and library operations. Independent reference harnesses use Transformers
to qualify outputs. Language choice alone does not establish performance; the
[benchmarking guide](benchmarking.md) defines the measurement boundary.

`runtime.py` owns host/GPU residency and selects an executor from validated model
configuration. `scheduler.py` owns bounded admission and continuous batching.
Completed or cancelled requests release slots; slow streams are subject to
backpressure. A model change at the FIFO queue head drains the current group
before activation. GPU-resident executors retain captured graphs. Host-cached
restoration normally constructs an executor again; compatible same-host layer
partitions can instead rebind existing weight buffers and retain graph addresses.
Failed rebinds are unpublished and closed before subsequent requests proceed.

Host cache limits account for retained model weights, not all transient loader
memory or process RSS. GPU admission includes path-specific weight, KV and
workspace estimates; estimates cannot guarantee every library allocation will
succeed. Fixed KV slots remain the default. Qualified CUDA dense decoders also
have an opt-in shared page pool. Prefix reuse copies bounded snapshots into
request-owned state; it does not share writable pages between requests.
See [performance tuning](performance.md) for paged KV, chunked prefill, prefix
reuse, prompt graphs and speculative decoding, including their individual
architecture and precision restrictions.

Dense rotary decoders share execution machinery where their semantics allow it.
Other families use dedicated executors for recurrent state, encoder-decoder
attention, mixture-of-experts routing, embeddings and reranking. Qwen3.5 owns
convolution/recurrent state alongside full-attention KV; its bounded still-image
path adds a vision encoder and image positions. These are checkpoint-qualified
paths, not arbitrary architecture acceptance. The [support matrix](support.md)
and [vision qualification](vision-qualification.md) specify current coverage.

Quantized execution dispatches by the actual format, device and qualified
geometry. Native Q4/Q8 conversion, selected GGUF layouts, AWQ and MXFP4 have
separate loaders and kernels. Host checkpoint representation need not match a
device-specific packed representation. For example, qualified Q4 layer stages
repack body weights in place for adjacent-nibble Tensor Core products while
retaining canonical host weights and output-head arithmetic. The opt-in A2
Q4 Marlin path replaces all qualified projections, including the output head,
and retains canonical host weights. Its executor owns the prepared layout and
shared workspace; admission includes layout overhead and temporary repacking
storage before GPU transfer. See
[quantization](quantization.md) for defaults, opt-in paths and numerical limits.

Apple execution combines MPS tensor operations with native Metal kernels.
Precision and available features depend on the model; CUDA graph/Triton paths
are not used on Apple. Dense, recurrent and bounded Qwen3.5 image paths have
separate M4 evidence. There is no automatic CPU inference fallback.

Multi-GPU execution currently uses a deliberately bounded Qwen2.5-0.5B layer
split. Same-host partitions share the normal residency manager and support
compatible swaps. The separate distributed service owns two persistent ranks,
typed tensor commands, bounded queues, heartbeats and failure shutdown. It does
not support model swaps inside a session. A2/3090 network proof uses NCCL sockets;
RDMA remains unverified. This is not general tensor/pipeline parallelism. See
[multi-GPU execution](multi-gpu.md) for exact limits and lifecycle evidence.

The following sections retain design requirements and direction. They do not
expand the implemented scope in the linked support matrix.

## Product contract

dllm owns model discovery, lifecycle, scheduling, and the user interface. A
single persistent API accepts a model name on each request. Loading another
model does not require restarting that API. Existing streams drain before their
model is evicted; a deadline or explicit cancellation may terminate them.

GPU execution must be measured on the target hardware. The selected direction
is NVIDIA first and an independent runtime: no vLLM runtime dependency. Reuse
mature kernel libraries where licensing permits instead of rebuilding every
matrix multiplication. The primary test device is the A2 on the lab VM. The A30 and desktop 3090
are also authorized for bounded tests that preserve their existing services;
the M4 laptop supplies Apple evidence. GPU experiments run sequentially so
measurements do not contend with other dllm validation jobs.

## Split responsibilities

- **Control service:** catalog, downloads, checksums, aliases, admission, memory
  budgets, lifecycle, and API/CLI. Never runs a per-token cross-process RPC.
- **Device workers:** persistent device contexts, weight loading, KV allocation,
  GPU execution, sampling, and batched delivery of output. A worker can replace
  a model executor while keeping device initialization alive where supported.
- **Execution plans:** keyed by device architecture, model architecture and
  revision, precision/quantization, parallelism, kernel version, and shape
  buckets. Separate compilation artifacts from live captured GPU graphs.

A native scheduler/worker boundary remains a direction to evaluate against
measured host overhead; the current scheduler is Python. CUDA and Metal already
have dedicated kernel paths. ROCm remains future work. Shared abstractions must
not force a slower GPU path, and additional architecture acceptance requires
independent numerical and serving qualification.

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
