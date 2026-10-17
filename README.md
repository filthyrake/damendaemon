# damendaemon

Damen's independent GPU inference runtime: fast generation and quick model switching.

**Follow development:** [project board](https://github.com/users/filthyrake/projects/3),
[current status and ordered next steps](https://github.com/filthyrake/damendaemon/issues/3),
[active work](https://github.com/filthyrake/damendaemon/issues?q=is%3Aopen%20label%3Astatus%3Aactive),
and [next tasks](https://github.com/filthyrake/damendaemon/issues?q=is%3Aopen%20label%3Astatus%3Anext).
Issue bodies carry the current status; comments and linked reports preserve the evidence.

damendaemon owns model execution, scheduling, sampling and residency. It uses
PyTorch/cuBLAS, fused Triton kernels, native CUDA kernels and Apple Metal;
it does not run vLLM, llama.cpp, Ollama or Transformers generation internally.
Transformers is used for validation and optional tokenizer adaptation.

This is an experimental runtime. CUDA and Apple Silicon paths, quantized
weights, image input and same-host two-GPU serving have qualified configurations.
Support and performance vary by checkpoint, precision and hardware. Cross-host
execution has experimental correctness evidence; RDMA remains unqualified.

Our [long-term targets](docs/direction.md) are Qwen3.8+, GLM-5.3/Flash and
DeepSeek V4 Flash, with 32K minimum context, useful multi-user concurrency,
and local/network GPU sharding across major vendors. These are goals, not
claims of current support.

## Quick start

Linux, Python 3.11+, an NVIDIA CUDA GPU and a compatible driver:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[runtime]'
export CUDA_VISIBLE_DEVICES=GPU-your-device-uuid

damendaemon run Qwen/Qwen3.5-0.8B --thinking off --prompt 'Explain CUDA graphs in one paragraph.'
damendaemon serve --model small=Qwen/Qwen3.5-0.8B
```

The fast CUDA path also needs a C compiler and matching Python development
headers. See [setup and Apple Silicon instructions](docs/getting-started.md).
The development server listens on `127.0.0.1:8088`; it has no authentication.
Registered models load on demand and restore from host memory after GPU eviction.

## Documentation

Start with the [documentation index](docs/README.md).

- [Serving, streaming and model switching](docs/serving.md)
- [Quantization, GGUF, AWQ and prepared weights](docs/quantization.md)
- [Performance tuning and cache policies](docs/performance.md)
- [Multi-GPU execution and its current limits](docs/multi-gpu.md)
- [Model-specific usage](docs/models.md) and [support matrix](docs/support.md)
- [Benchmarking and measured results](docs/benchmarking.md)
- [Runtime architecture](docs/architecture.md)

Performance is a product requirement. We report measured first-token latency,
decode speed and concurrent throughput separately, with independent correctness
checks and explicit qualification limits. See the linked evidence before
extrapolating a result to another model or device.

The [parity milestones and acceptance gates](https://github.com/filthyrake/damendaemon/issues/3)
define the ordered work; [GitHub Issues](https://github.com/filthyrake/damendaemon/issues)
are the bug and task tracker. [AMD/ROCm bring-up](docs/amd.md) includes bounded Qwen3.5
and opt-in Qwen3.8-27B Q8 text correctness and lifecycle evidence; practical long-context
and competitive performance qualification remain open in [the Qwen tracker](https://github.com/filthyrake/damendaemon/issues/55). See [project license](LICENSE.md) and the separately retained
[Marlin license](third_party/marlin/LICENSE).
