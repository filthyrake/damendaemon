# dllm

Damen's LLM runtime: fast GPU inference with effortless model switching.

**Current state:** an independent GPU inference prototype, verified on an A2
with Qwen2.5, Qwen3, Qwen3.5 text models, and SmolLM2, plus an experimental
Apple Silicon MPS path. It implements its own forward pass, static KV
cache, seeded sampling, CUDA graphs, continuous batching, a streaming API, and
GPU/host model residency. PyTorch/cuBLAS and dllm's fused Triton kernels provide
GPU execution; dllm does not depend on vLLM, llama.cpp, Ollama,
or Transformers generation. Transformers is used for validation and optional structured-output tokenizer adaptation; model execution remains independent.

Additional qualified checkpoints include Llama 3.2, Gemma 3 text, Phi-3
mini, Mistral 7B, StarCoder2 3B, GPT-2, Pythia (GPT-NeoX), BLOOM, MiniLM sentence embeddings, and CUDA OLMoE. Exact revisions, numerical gates, and remaining limits are recorded in
[architecture evidence](docs/architecture-parity.md). The [current support matrix](docs/support.md) summarizes architecture and serving boundaries.

The server selects models from each request without restarting. GPU-resident
models retain their execution graphs; evicted models restore from the host cache.
This remains an early runtime with deliberately narrow model support.

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
repository revision.

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
Apple GPU, with no automatic CPU fallback. Metal performance parity remains open. The same scheduler and model residency interface apply. Hybrid MPS
execution is not yet qualified. See [hardware-specific evidence](docs/hybrid-and-mps.md).

For lower device-weight memory and faster decoding, set
`DLLM_METAL_WEIGHT_DTYPE=float16` with the native Metal backend. Arithmetic and
KV remain FP32; this differs from the unqualified FP16-compute tensor path.
Prefill currently expands matrix weights to FP32 temporarily, which increases
first-output latency. FP32 weight storage remains the default.

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

For raw text or code generation, `/v1/completions` accepts `prompt` as a string
or a nonempty array of token IDs. It shares sampling, stop sequences, streaming,
batching, and model switching with chat. Supported base checkpoints can load
without a chat template; chat requests to those models are rejected. Text prompts
use the tokenizer's normal special-token processing; token-ID prompts are used
exactly as supplied. One prompt/completion per request is supported; prompt
batches, echo, and logprobs are not yet implemented.
The CLI equivalent is `dllm run MODEL --raw --prompt 'text to continue'`.

StarCoder2 3B is qualified at 2048 context with original FP32 weights on A2 and
Apple Silicon (`--host-gib 16`), and with optional CUDA Q8 weight storage.
Mistral 7B Instruct v0.1 is qualified on A2 with Q8 storage through its 4096-token
window boundary. See the architecture evidence for exact revisions and limits.
Enable experimental Q8 on-load conversion with `DLLM_QUANTIZATION=q8_0`; it
currently requires CUDA and does not import GGUF/AWQ/GPTQ files.

```sh
curl http://127.0.0.1:8088/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"small","prompt":"def binary_search(values, target):\n    ","max_tokens":64,"stream":true}'
```

Qwen3 and Qwen3.5 thinking default to their model templates. Use `dllm run Qwen/Qwen3-1.7B
--thinking off --prompt 'What is 7 times 8?'`, or pass
`"chat_template_kwargs": {"enable_thinking": false}` in the API request to disable
it. Explicit thinking controls are rejected for other architectures. Thinking
text is returned as model content; it is not parsed into a separate field.

Change `model` to `medium` on the next request. `--gpu-models 1` forces eviction
and host restoration when switching; `--gpu-models 2` keeps both resident if they
fit. Host weights default to two models with an 8 GiB retained-weight budget.
The budget excludes transient loading buffers and library overhead. GPU capacity
is checked conservatively; allocation failures return an error without publishing
a partially loaded executor. There is no automatic GPU-budget tuning yet.

Implemented: dense unquantized Qwen2, Qwen3, Llama, Phi-3, and the
dense Qwen3.5 text backbone; text chat templates, greedy or sampled decoding, and optional
streaming usage. CUDA uses FP16 weights/KV. Qwen3 uses FP32 residuals and fused
elementwise intermediates for accuracy, while its matrix multiplies remain FP16.
Phi-3 adds packed projections, sliding attention and native 96-wide heads; its
CUDA path retains FP16 matrix/KV storage with FP32 residuals and final logits.
The pinned Phi-3 mini 4K checkpoint passes the strict 4K CUDA replay; Apple has
a 4K single-slot qualification. LongRoPE Phi variants remain unsupported.
Sampling accepts finite `temperature >= 0`, `top_p` in `(0, 1]`, nonnegative integer
`top_k` (zero disables it), and an optional unsigned 64-bit `seed`. Temperature
defaults to zero and retains the greedy fast path. Seeds belong to requests;
batching and cancellation do not share RNG state. Exact outputs can still vary
across devices, kernels, and precision. Unsupported fields such as tools and
multimodal inputs are rejected.
Qwen3.5 has a separate hybrid executor with FP32 recurrent state, causal
convolution, gated attention, partial RoPE, and FP32 vocabulary accumulation.
Vision, MoE, dynamic rotary scaling, and quantization remain unsupported.
Gemma 3 text has a separate full-precision correctness path with alternating
sliding/full attention, GELU gating and its own normalization; the pinned 270M
checkpoint passes CUDA and Metal reference replay, including 511/512/513-token
window boundaries. CUDA uses FP16 weight storage with native FP32 matrix-vector
arithmetic and windowed decode attention; arithmetic and KV remain FP32.
`DLLM_GEMMA_WEIGHT_DTYPE=float32` selects full storage. Prefill optimization remains open. This does not imply support for multimodal Gemma. Qwen3.5 checkpoint qualification does not imply all newer Qwen
variants are supported. See the [model validation matrix](docs/model-matrix.md):
Qwen3-1.7B and SmolLM2-360M pass the expanded tests; Qwen3-0.6B remains provisional
because two long-context positions exceed the logit-error limit.
Static linear and Llama 3 rotary scaling are now supported; dynamic/YaRN/LongRoPE
variants remain unsupported. The pinned Llama 3.2 1B checkpoint passes the A2
8192-context gate. See [architecture qualification](docs/architecture-parity.md).
Chat requests also accept `stop` (one string or up to four strings, each 1–256
characters). Matching stop text is excluded from streamed and complete responses;
partial matches are withheld until they resolve. `max_completion_tokens` is an
alias for `max_tokens`; specify only one.

Prompt plus requested output must fit `--context`. KV memory is reserved per
slot at that capacity. Fused attention reads the live prefix and skips unused
context blocks; larger configured contexts still add launch/merge overhead.

The server continuously batches up to four requests for one model by default;
use `--batch-size 1..8` to choose capacity. Different prompt lengths have separate
KV slots. Dense decoders pack available prompts into shared linear layers;
hybrid prompts currently prefill separately. Model groups drain
before switching; a waiting different model prevents later requests from jumping
ahead of it. Prefill is still unchunked, so a long prompt can delay active decode.
Up to eight requests can be admitted, with a 120-second queue wait/backpressure
limit; HTTP connections are separately capped at 16.
Disconnects close the generation iterator once a socket write detects them.
The loopback development server has no authentication and is not a public gateway.

## Verified results

Profiling identified excessive small-kernel launches in the first executor.
Fused normalization, RoPE/cache writes, activation, and attention reduced decode
from 1,364 to 274 kernels per token. Continuous batching now shares weight reads
across requests. Performance is measured on the A2 against the same immutable
FP16 model snapshots, with independent-reference correctness checks. The final
0.5B HTTP run delivered 139 tokens/s after first output (vLLM: 149), and four
concurrent requests delivered 433 aggregate output tokens/s including prefill
(vLLM: 497). These are short-workload measurements, not a broad parity claim.

See the [hybrid and Apple Silicon proof](docs/hybrid-and-mps.md),
[new-model and longer-context results](docs/model-matrix.md), and
[fused and batched runtime evidence](docs/batching-proof.md) for current
HTTP comparisons, correctness gates, model switching, and limitations. The
[first runtime evidence](docs/runtime-proof.md) preserves the original baseline.

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

Optional constrained JSON responses are enabled by installing `dllm[structured]`.
Both completion routes accept `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":...}}`.
The supported schema subset includes object properties/required/additionalProperties,
arrays with an item schema, primitive types, and primitive enum/const values.
Unknown keywords are rejected. Schemas are limited to 16 KiB, 256 nodes and depth
16; property order follows the schema. A completion that exhausts its token
budget has finish reason `length` and can contain incomplete JSON.

Grammar masks apply before sampling, with independent state for each request.
Structured responses reject custom stop strings and explicit thinking; Qwen3/
Qwen3.5 chat thinking is automatically disabled. Plain requests keep the existing
fast path. The optional XGrammar compiler and mask kernels are reused; dllm
still executes the model itself. Apple uses a native PyTorch/Metal bitmask path.

JSON formatting permits at most eight whitespace characters between grammar
elements, preventing unbounded whitespace-only generation. String contents are
not subject to this formatting limit.

For accurate CUDA Qwen3 arithmetic, set `DLLM_QWEN3_PRECISION=float32`.
This retains compact FP16 matrix storage but uses FP32 operations and KV; it
clears the Qwen3-0.6B long-context numerical gate. The default `standard` path
remains faster for batched decoding but retains that checkpoint's documented
accuracy qualification gap. See the measurements in the architecture report.
The accurate Qwen3 mode uses compensated Tensor Core products for small batches:
FP32 activations are split into scaled high/residual FP16 components, with FP32
accumulation. Both tested Qwen3 sizes pass independent FP32-reference gates;
this is qualified numerical approximation, not bitwise FP32 equivalence.

You can start `dllm serve` with no models and manage its registry without a
restart. Run these commands on the server's host:

```sh
dllm add small Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
dllm load small
dllm models
dllm unload small                 # retain the RAM copy
dllm unload small --release-host  # release GPU and RAM copies; keep disk files
```

`add` also accepts an existing local snapshot directory. Downloads happen in the
CLI before registration. Commands default to `http://127.0.0.1:8088`; use
`--server` to select another endpoint. Snapshot paths must exist on that server's
host. Alias registration is idempotent for the same path; an existing alias
cannot silently change to different weights. The registry is process-local, so
use startup `--model` entries when you want aliases restored after a restart.

The dllm HTTP extensions are `GET /v1/models/status` and
`POST /v1/models/register`, `/v1/models/load`, `/v1/models/unload`.
POST bodies use `model`; registration also requires `path`, and unloading accepts
boolean `release_host`. Commands share bounded admission and wait for active
model groups to finish. Ordinary generation still loads registered models on
demand. These extensions are separate from OpenAI's completion protocol.

Sentence embeddings are available through `POST /v1/embeddings`. Register the
pinned `sentence-transformers/all-MiniLM-L6-v2` snapshot at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` like any other model, then send:

```json
{"model":"minilm","input":["A dog plays outside.","A puppy runs in the park."],"encoding_format":"float"}
```

The independent BERT executor supports masked mean pooling and normalized
384-dimensional vectors on CUDA and Apple Silicon. Inputs can be strings or
explicit token-ID sequences, with up to 64 inputs per request. Float and
little-endian FP32 base64 output are supported. `dimensions`, if supplied, must
be 384. Inputs exceeding 256 tokens (including special tokens) are rejected,
not silently truncated. This is qualification of the pinned MiniLM checkpoint,
not all SentenceTransformers models. Encoder quantization is unsupported; register this encoder with
`--quantization none` when the server default is Q8. Requests share
the model-residency scheduler with chat and completion workloads. See
[encoder evidence](docs/architecture-parity.md#sentence-embeddings) for precision,
performance, and current batching limits.

Both completion routes accept `presence_penalty` and `frequency_penalty` in
[-2, 2], positive `repetition_penalty` (default 1), and `logit_bias` mapping up to
1024 token IDs to finite biases in [-100, 100]. Bias IDs must belong to the
selected model's vocabulary. Presence/frequency penalties count generated
tokens only; repetition considers the prompt and generated tokens. Processing
order is repetition, frequency/presence, bias, grammar mask, then temperature
and top-k/top-p filtering. These controls apply to greedy requests too. Request
histories are isolated; default requests retain batched argmax and do not track
penalty counts. Optional controls currently use device tensor operations and
incur extra launches rather than a fused penalty kernel.


Quantization can be selected per alias at registration:

```sh
dllm add chat /path/to/chat/snapshot --quantization q8_0
dllm add encoder /path/to/minilm/snapshot --quantization none
```

`POST /v1/models/register` accepts the same `quantization` field (`none`,
`q8_0`, or `q4_0`), and `/v1/models/status` reports each alias's policy. Omission uses the
server's `DLLM_QUANTIZATION` default captured at startup. Policies are part of
alias identity and remain stable across eviction/reload; register another alias
for a different precision. Q8 still requires CUDA. A Q8 decoder and an
unquantized encoder can share one server and its GPU/host residency budgets.


GPT-2 learned-position decoding is qualified on A2 and Apple M4 with FP32
weights/arithmetic/KV through its complete 1024-token position table. Use the
pinned `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` with `--raw` or `/v1/completions`.
This base model has no chat template. GPT-2 quantization is not yet supported;
register it with `--quantization none` when the server default is Q8.

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

### Experimental packed four-bit weights

`dllm add compact /path/to/snapshot --quantization q4_0` enables native CUDA
Q4_0 projection weights with FP16 block scales. Input embeddings and norms keep
their existing precision. `DLLM_QUANTIZATION=q4_0` sets the startup default;
per-model policies can mix Q4, Q8 and unquantized models in the same server.

Independent model gates pass Qwen2.5-0.5B, accurate Qwen3-0.6B and
Llama3.2-1B on A2, along with live model-switching tests. The loader currently
allows dense Qwen2/Qwen3/Llama/Phi-3/Mistral only; acceptance alone is not a
checkpoint qualification. Four-bit quantization is lossy: on our small authored
171-token Qwen2 corpus, next-token agreement with the original weights was
74.3% for Q4 versus 95.3% for Q8. This is not a broad quality evaluation.
Q4 remains opt-in and CUDA-only. This is on-load conversion from safetensors;
GGUF, AWQ and GPTQ checkpoint import are still unsupported.

### Static YaRN configurations

Dense Qwen2/Qwen3/Llama/Mistral checkpoints may use static `rope_scaling` with
`rope_type: yarn`, `factor`, and optional `original_max_position_embeddings`,
`beta_fast`, `beta_slow` and `attention_factor`. On CUDA this selects FP32
arithmetic/KV with FP16 stored matrices to meet the numerical gate, so budget
for the larger KV cache. MPS retains its FP32 execution policy. Full and chunked
model execution is verified on A2/M4 using an explicit Qwen2 configuration
fixture; this does not establish trained 128K-context quality. Dynamic NTK and
LongRoPE remain unsupported.

### GPT-NeoX base models

The independent GPT-NeoX decoder supports partial rotary embeddings and parallel
residual blocks. Pinned Pythia-160M is qualified on A2 and Apple M4 through 2048 context
and the raw completions API. It uses FP32
storage/arithmetic/KV. Use `dllm run PATH --raw --prompt 'Once upon a time'` or
`/v1/completions`; the base checkpoint has no chat template. Quantization,
chunked prefill and prefix reuse are not yet supported for this architecture.

### Optional CUDA C++ Q4 decode

A separately built native kernel improves single-request Mistral Q4 decode on
A2. Build for your GPU architecture (86 in this A2 example):

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dllm-q4 -j2
export DLLM_Q4_CUDA_LIBRARY="$PWD/build/libdllm-q4.so"
```

Set the library path before starting the server. The runtime checks its ABI
and uses it only for qualified FP16 single-row shapes; other shapes retain the
Triton path. CUDA graph capture uses the caller's current stream. The measured
Mistral Q4 result is 22.53 tokens/s versus 20.06 with the paired Tensor Core
path alone, still below the matched llama.cpp result of 34.20. There is no
runtime compilation requirement when this option is unset.

### Prepared weight cache

Set `DLLM_WEIGHT_CACHE=/path/to/trusted/cache` before starting dllm to reuse
prepared Q4/Q8 weights across processes and cold model activations. This is
currently available for dense Qwen2, Qwen3, Llama, Phi-3 and Mistral. The first
load converts weights and writes a safetensors file; subsequent loads reuse it.
GPU-resident and host-resident switching continue to use their existing paths.

Cache keys include the preparation version, validated configuration, precision,
and source shard paths and file metadata. This is local file identity, **not a
cryptographic checkpoint digest**. Use trusted checkpoint and cache directories.
Missing, truncated or incompatible entries are rebuilt; publication is atomic.
Cache write failures warn and leave inference available. Changing a source during
loading is an error. There is no automatic disk eviction: budget roughly one
extra prepared-weight copy per checkpoint and precision, and remove obsolete
cache files when desired. No files are created unless this option is set.

### BLOOM / ALiBi

Pinned BLOOM-560m is qualified through 2048 context on A2 and Apple M4. Its
independent decoder implements ALiBi attention, embedding LayerNorm, head-major
packed QKV and the tied output projection. It uses FP32 weights/arithmetic/KV;
CUDA decode applies ALiBi directly inside the slotted attention kernel. Prefill
bounds additive-mask workspace by processing 128 query positions at a time.
This is internal attention tiling, not scheduler-level chunked prefill.

Use raw completions for the base checkpoint, which has no chat template.
Quantization, prompt-prefix reuse and interleaved prefill are not yet supported
for BLOOM. Acceptance of another BLOOM configuration does not qualify its
checkpoint or an extended context. See [architecture evidence](docs/architecture-parity.md).

### Shared paged KV cache (CUDA, opt-in)

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
and Mistral Q4 pass chunked reference checks; Llama Q4 currently exceeds the
1% numerical-error gate on one chunked check and is not qualified for that mode.
Prefix snapshots remain
unsupported in combination, as do other architectures and Apple Silicon.
It does not quantize KV or share pages between different prompts. Full prefill
uses fused current-token K/V outputs; offset chunks gather prior pages, and
decode reads pages directly. Reducing
the pool saves allocated GPU memory but may reduce simultaneous admission for
long requests. See [paged-cache evidence](docs/evidence/paged/README.md).
