# dllm

Damen's LLM runtime: fast GPU inference with effortless model switching.

**Current state:** architecture, a working streaming endpoint benchmark, and a
native CUDA/cuBLAS hardware probe verified on an NVIDIA A2.
There is no dllm inference runtime yet, and no measured performance claim.

The target experience is `dllm run <model>` and a persistent server where the
request's model selects what to run. Those commands are design targets, not
implemented commands. Users should not need to restart a server or tune dozens
of flags to change models.

## Design priorities

1. Low time to first token and fast decoding for interactive use.
2. High aggregate throughput under concurrent load without unbounded latency.
3. Fast switching among GPU-resident, RAM-cached, and disk-cached models.
4. Explicit memory budgets, cancellation, and predictable behavior under pressure.
5. GPU-specific optimized execution behind a stable model-management interface.

See [architecture](docs/architecture.md) for the execution and residency design.

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

GitHub Issues are the project tracker once a remote repository is configured.

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
[the hardware evidence](docs/a2-hardware.md).
