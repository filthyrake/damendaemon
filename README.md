# dllm

Damen's LLM runtime: fast GPU inference with effortless model switching.

**Current state:** a working independent NVIDIA inference prototype, verified on
an A2 with Qwen2.5 0.5B and 1.5B. It implements its own forward pass, static KV
cache, greedy decoding, CUDA graphs, streaming API, and GPU/host model residency.
PyTorch provides the GPU kernels; dllm does not depend on vLLM, llama.cpp, Ollama,
or Transformers generation. Transformers is only a validation dependency.

The server selects models from each request without restarting. GPU-resident
models retain their execution graphs; evicted models restore from the host cache.
This is an early, single-request executor, not yet a general-purpose replacement.

## Run a model

Linux, Python 3.11+, NVIDIA CUDA GPU and a compatible driver. Install the CUDA
build of PyTorch into a fresh environment before installing dllm's runtime extra:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[runtime]'

export CUDA_VISIBLE_DEVICES=GPU-your-device-uuid
dllm run Qwen/Qwen2.5-0.5B-Instruct --prompt 'Explain CUDA graphs in one paragraph.'
```

`run` downloads the model if needed. `dllm pull OWNER/REPOSITORY` downloads
without requiring a GPU and prints the resolved immutable revision and local
snapshot path. Use that path for offline runs, or `run --revision SHA` to pin a
repository revision.

The explicit GPU UUID prevents accidentally using another busy device. Obtain
UUIDs with `nvidia-smi -L`. No GPU is needed for downloading or HTTP benchmarking.

## Serve and switch models

```sh
dllm serve --model small=Qwen/Qwen2.5-0.5B-Instruct \
  --model medium=Qwen/Qwen2.5-1.5B-Instruct --gpu-models 2 --context 1024
```

Local snapshot paths also work in `--model`. Repository names are resolved and
downloaded before the server starts. The API binds to `127.0.0.1:8088`.
Use `/v1/models` and `/v1/chat/completions`:

```sh
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"small","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0,"max_tokens":32,"stream":true}'
```

Change `model` to `medium` on the next request. `--gpu-models 1` forces eviction
and host restoration when switching; `--gpu-models 2` keeps both resident if they
fit. Host weights default to two models with an 8 GiB retained-weight budget.
The budget excludes transient loading buffers and library overhead. GPU capacity
is checked conservatively; allocation failures return an error without publishing
a partially loaded executor. There is no automatic GPU-budget tuning yet.

Supported: dense unquantized Qwen2, standard RoPE, text chat templates, FP16,
greedy decoding, optional streaming usage. `temperature` must be zero. Unsupported
fields are rejected, including tools, multimodal inputs, and alternate sampling.
Prompt plus requested output must fit `--context`. Decode attends over that fixed
capacity, so larger contexts can cost speed even for short requests.

Requests run one at a time across all resident models. Up to eight can be admitted,
with a 120-second queue wait limit; HTTP connections are separately capped at 16.
Disconnects close the generation iterator once a socket write detects them.
The loopback development server has no authentication and is not a public gateway.

## Verified results

On the A2, the 0.5B executor measured about 110 decode tokens/s with CUDA graphs,
versus 37–38 without them; 1.5B measured about 43 versus 31. Both models matched
192 combined greedy token checks against an independent FP16 reference. The
real API passed repeated model switching, streaming, concurrent request
serialization, disconnect recovery, and failed-load recovery.

A matching 0.5B FP16 HTTP workload measured approximately 106 tokens/s after
first output for dllm, 66 for the tested llama.cpp image, and 150 for vLLM 0.30.0.
dllm has not reached vLLM performance parity. This is one short, single-request
workload, not a general speed ranking. See
[runtime evidence](docs/runtime-proof.md) for revisions, settings, raw measurements,
correctness tolerances, switching results, and limitations.

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
