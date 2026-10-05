# Getting started

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Install dllm, select a device explicitly, and pin model revisions for repeatable runs.

## Run a model

Linux, Python 3.11+, NVIDIA CUDA GPU and a compatible driver. Install the CUDA
build of PyTorch into a fresh environment before installing dllm's runtime extra.
The fast Triton path also needs a C compiler and matching Python development
headers for its CUDA driver helper:

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
repository revision. Pull selects the shards named by the root safetensors index
(or the root `model.safetensors`) and preserves model metadata and licenses;
alternate weight copies elsewhere in the repository are not downloaded.

`DLLM_KERNELS=auto` (default) selects fused Triton kernels when available and
falls back to PyTorch if its import or compiler setup is unavailable. Set
`DLLM_KERNELS=triton` to require the fast path, or `DLLM_KERNELS=torch` to use the
reference tensor path. Non-streaming responses report the selected backend in
`dllm_timing.kernel_backend`.

The explicit GPU UUID prevents accidentally using another busy device. Obtain
UUIDs with `nvidia-smi -L`. No GPU is needed for downloading or HTTP benchmarking.

For optional parallel hybrid prefill, install `.[hybrid]` and set
`DLLM_HYBRID_PREFILL=fla`. This uses the kernel-only FLA package for prompts of
64 tokens or more; dllm still owns model execution, state, scheduling, and
switching. The default `scan` path has no FLA dependency. Both paths are checked
against independently generated FP32 reference logits. First-use compilation
can be substantial; warm measurements exclude it.

## Apple Silicon (experimental)

Install `.[runtime]` in a native arm64 Python environment on macOS and select MPS:

```sh
dllm run Qwen/Qwen2.5-0.5B-Instruct --device mps --prompt 'What is 2 + 2?'
dllm serve --device mps --model small=Qwen/Qwen2.5-0.5B-Instruct \
  --model other=HuggingFaceTB/SmolLM2-360M-Instruct
```

The dense MPS path uses FP32 weights, activations, and KV. On PyTorch 2.13,
`auto` selects dllm's native Metal kernels for normalization, activation,
small-batch matrix operations, rotary/cache writes, and decode attention.
`DLLM_KERNELS=metal` requires them; `DLLM_KERNELS=torch` selects the tensor
baseline. CUDA graphs and Triton are disabled on MPS. Execution stays on the
Apple GPU, with no automatic CPU fallback. Metal performance parity remains open.
The same scheduler and model residency interface apply. Dense text details are in
[hardware-specific evidence](hybrid-and-mps.md); the separately qualified
Qwen3.5 hybrid image path and its FP16/Q8 limits are in
[vision qualification](vision-qualification.md).

For lower device-weight memory and faster decoding, set
`DLLM_METAL_WEIGHT_DTYPE=float16` with the native Metal backend. Arithmetic and
KV remain FP32; this differs from the unqualified FP16-compute tensor path.
Prefill currently expands matrix weights to FP32 temporarily, which increases
first-output latency. FP32 weight storage remains the default.
