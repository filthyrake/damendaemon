# Benchmarking and evidence

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Compare identical revisions, precision, context, output lengths, and hardware. CPU tests establish behavior, not GPU performance.

The [fixed performance scorecard](https://github.com/filthyrake/dllm/issues/54)
defines the next required workloads and competitive thresholds. These are
targets, not results. Functional, numerical, lifecycle and performance gates
are recorded separately; a missing or failed comparison never counts as parity.

For a warm result, initialize every execution shape the measured workload can
reach, not just one request at each concurrency. Arrival timing can make the
first round process prompts singly and expose a packed shape only during timed
rounds. Record actual group sizes and warmup cost; report first-use shape latency
separately. The [Q4 network comparison](evidence/quantization/cuda/q4-arithmetic/README.md#all-reachable-prompt-shapes-warm)
shows why this distinction matters. Generic `dllm bench` does not do that warmup
for you.

## Qualified baseline results

Profiling identified excessive small-kernel launches in the first executor.
Fused normalization, RoPE/cache writes, activation, and attention reduced decode
from 1,364 to 274 kernels per token. Continuous batching now shares weight reads
across requests. Performance is measured on the A2 against the same immutable
FP16 model snapshots, with independent-reference correctness checks. The final
0.5B HTTP run delivered 139 tokens/s after first output (vLLM: 149), and four
concurrent requests delivered 433 aggregate output tokens/s including prefill
(vLLM: 497). These are short-workload measurements, not a broad parity claim.

See the [hybrid and Apple Silicon proof](hybrid-and-mps.md),
[new-model and longer-context results](model-matrix.md), and
[fused and batched runtime evidence](batching-proof.md) for current
HTTP comparisons, correctness gates, model switching, and limitations. The
[first runtime evidence](runtime-proof.md) preserves the original baseline.

## Measure existing servers

Python 3.11 or newer; no third-party dependencies required:

```sh
python3 -m dllm bench --base-url http://localhost:8000/v1 \
  --models model-a --requests 16 --concurrency 4 --max-tokens 128 \
  --label baseline > baseline.json

# Sequential A -> B -> A requests measure user-visible model transitions.
# The target server must already support routing/loading these models.
python3 -m dllm bench --base-url http://localhost:8080/v1 \
  --models model-a model-b model-a --requests 3 --concurrency 1 \
  --label switching > switching.json

python3 -m unittest discover -s tests -v
```

Set `DLLM_API_KEY` if the endpoint needs authentication. The key is not saved in
results. Use `--prompt-file` for a repeatable workload and `--label` to identify
hardware, model revision, quantization, and server configuration. Prompts are
represented by a hash in the report, not saved as text.

Reports include each request, failures, wall-clock throughput, first visible
output latency, and output-token counts when the server reports them. Streaming
chunks are **not** counted as tokens. Missing usage means token throughput is
unknown. First visible output can lag actual first-token computation, especially
with reasoning models or server buffering. Full details: `python3 -m dllm --help`.

This is a closed-loop load test: at most `--concurrency` requests are in flight.
It does not measure open-loop arrival-rate capacity, GPU utilization, or internal
weight-transfer timing. Switching timings include everything the server does
before responding; resident versus RAM versus disk state must be controlled and
recorded externally. No warmup or eviction is performed automatically.


## CUDA hardware probe

This standalone probe measures pinned-host-to-device transfers and FP16 matrix
multiplication, validates results, and releases its resources on exit. It refuses
to run without exactly one GPU UUID selected. It is not a model runtime or an
LLM tokens-per-second benchmark. Peak allocations are under 512 MiB GPU memory
and approximately 256 MiB pinned host memory, excluding CUDA/library overhead.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j2
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid ./build/dllm-cuda-probe
```

Alternatively, with an installed CUDA toolkit:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 native/cuda_probe.cu -lcublas -o /tmp/dllm-cuda-probe
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid /tmp/dllm-cuda-probe
```

The first A2 measurements and their limitations are in
[the hardware evidence](a2-hardware.md).
