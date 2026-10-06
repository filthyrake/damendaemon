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
dllm run Qwen/Qwen3.5-0.8B --thinking off --prompt 'Explain CUDA graphs in one paragraph.'
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
dllm run Qwen/Qwen3.5-0.8B --device mps --thinking off --prompt 'What is 2 + 2?'
dllm serve --device mps --model small=Qwen/Qwen3.5-0.8B \
  --model other=Qwen/Qwen3.5-2B
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

## AMD practical-model preview

The AMD branch has a bounded, opt-in Qwen3.8-27B Q8 text configuration on
HOMER's Radeon 8060S. Use the existing tested ROCm environment described in
[AMD qualification](amd.md); the CUDA installation commands above do not apply.
This preview supports 512 total prompt-plus-output tokens, up to four request
slots, and no graphs. It is not the planned 32K daily-use configuration.

Download the pinned snapshot, then use the local path printed by `pull`:

```sh
dllm pull Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
DLLM_KERNELS=triton DLLM_QUANTIZATION=q8_0 \
DLLM_HYBRID_PRECISION=compensated DLLM_ROCM_MEMORY_GIB=48 \
dllm serve --device rocm --model chat=/path/printed/by/pull \
  --context 512 --batch-size 4 --no-graph --host-gib 40
```

The host-weight budget must exceed the approximately 27.73 GiB of Q8 weights;
its default 8 GiB is insufficient. The device budget includes execution state
and workspace and must also preserve system headroom on a shared-memory GPU.
These values describe HOMER, not all AMD devices. First loading includes
on-load quantization and compilation; it can take substantially longer than
subsequent requests.

Send a short request using the ordinary API:

```sh
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Explain what a GPU does in two sentences."}],"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"stream":true}'
```

See [serving and model management](serving.md) for status, switching and release.
For the independently checked Q4 variant on HOMER, replace `DLLM_QUANTIZATION=q8_0`
with `DLLM_QUANTIZATION=q4_0 DLLM_EXPERIMENTAL_HYBRID_Q4=1`, retaining the
other bounds above. Packed weights use 15.80 GiB. The numerical and HTTP gates
pass, but this is not original-weight quality or a faster-than-Q8 claim.
32K context, practical multimodal support and other-device qualification remain
in [the current roadmap](https://github.com/filthyrake/dllm/issues/3).
