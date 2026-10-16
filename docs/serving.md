# Serving and model management

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Use one server for registered model aliases, streaming requests, and explicit residency control.

Request JSON rejects duplicate object fields and nonfinite numbers before dispatch.
Excessive decoder nesting returns an invalid-request response.

HTTP JSON request bodies are limited to 1 MiB and 30 seconds of total read time.
Partial progress does not extend the body deadline. Expiry closes the connection
before runtime dispatch and runs admission cleanup. This is a body-read bound,
not a total header-read or inference deadline; the socket's previous inactivity
timeout is restored before inference.

## Serve and switch models

```sh
dllm serve --model small=Qwen/Qwen3.5-0.8B \
  --model medium=Qwen/Qwen3.5-2B --gpu-models 2 --context 1024
```

Local snapshot paths also work in `--model`. Repository names are resolved and
downloaded before the server starts. The API binds to `127.0.0.1:8088`.
Use `/v1/models` and `/v1/chat/completions`:

```sh
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"small","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":32,"stream":true}'
```

## Repeatable model configuration

Use `dllm serve --model-config models.json` to reuse model aliases and precision
across server restarts. For example, save this versioned JSON file:

```json
{
  "version": 1,
  "models": {
    "chat": {
      "repository": "Qwen/Qwen3.8-27B",
      "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      "quantization": "q4_0"
    }
  }
}
```

Keep the device, context, experimental opt-ins and memory settings from the
[qualified getting-started command](getting-started.md); replace its `--model`
argument with `--model-config models.json`. The manifest does not enable a new
architecture or change hardware admission. Every model requires explicit
`quantization` (`none`, `q8_0` or `q4_0`), independent of the environment default.

Repository sources require an immutable 40-character revision. For a GGUF
repository, add the exact `filename`; this reuses the ordinary single-file
download path. Alternatively, use `"path": "snapshots/my-model"` instead of
`repository` and `revision`. Relative local paths resolve from the manifest's
directory, not the shell's current directory. Local files remain user-owned;
the manifest does not hash them or prevent their modification.

An optional `chat_template` contains the actual override text, preserving its
identity without depending on another editable template file. Omit it to use
the pinned checkpoint's template. Unknown/duplicate fields, ambiguous sources
and conflicting `--model` or `--chat-template` options are rejected. Downloads
use the existing cache. Resource limits and environment settings remain explicit
CLI/service settings. The server never rewrites the file automatically.

After adding models to a running server, save its current registry explicitly:

```sh
dllm models --save-config models.json
# After stopping the server, reuse the saved models with the same device/budgets:
dllm serve --model-config models.json --device cuda --context 512 --host-gib 40
```

Keep any required experimental opt-ins and other options from your original
command. Export captures resolved local paths, precision and template text;
it does not load or unload models, wait for inference, or save GPU/environment
settings. Paths belong to the server's filesystem, so the file is intended for
restarting on that host. `--server URL` selects an existing server. The read-only
API is `GET /v1/models/config`. The CLI replaces the destination atomically;
failed requests or writes preserve its previous content. An empty registry or
export larger than the manifest's 1-MiB limit is rejected. Registrations added
after saving require another explicit export to survive restart.

For raw text or code generation, `/v1/completions` accepts `prompt` as a string
or a nonempty array of token IDs. It shares sampling, stop sequences, streaming,
batching, and model switching with chat. Supported base checkpoints can load
without a chat template; chat requests to those models are rejected. Text prompts
use the tokenizer's normal special-token processing; token-ID prompts are used
exactly as supplied. One prompt/completion per request is supported; prompt
batches, echo, and raw-completion logprobs are not yet implemented.
Chat log probabilities are described [below](#token-log-probabilities).
The CLI equivalent is `dllm run MODEL --raw --prompt 'text to continue'`.

StarCoder2 3B is qualified at 2048 context with original FP32 weights on A2 and
Apple Silicon (`--host-gib 16`), and with optional CUDA Q8 weight storage.
Mistral 7B Instruct v0.1 is qualified on A2 with Q8 storage through its 4096-token
window boundary. See the architecture evidence for exact revisions and limits.
Enable experimental Q8 on-load conversion with `DLLM_QUANTIZATION=q8_0`; it
supports native Metal for Qwen2, Qwen3, Llama and Mamba-1/2; other families require CUDA.
See [GGUF and AWQ import scope](quantization.md) for supported packed formats.
GPTQ remains unsupported.

```sh
curl http://127.0.0.1:8088/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"small","prompt":"def binary_search(values, target):\n    ","max_tokens":64,"stream":true}'
```

## Thinking and chat behavior

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

While generation owns the worker, status waits at most one second (or a shorter
configured queue timeout) before returning HTTP 503. Other queued management
commands use the configured queue timeout; expired queued commands are removed
and will not execute later. A command already executing waits for its actual
result. Model changes remain ordered behind active generation; status is not a
live progress snapshot. `/health` remains available independently.

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
across devices, kernels, and precision. Unsupported request fields and content
types are rejected; image inputs require the experimental opt-in below.
Qwen3.5 has a separate hybrid executor with FP32 recurrent state, causal
convolution, gated attention, partial RoPE, and FP32 vocabulary accumulation.
Q8 text projections are supported on the qualified CUDA path. Hybrid MoE and
dynamic rotary scaling remain unsupported. Still-image serving is experimental
and opt-in for the qualified Qwen3.5-0.8B geometry on NVIDIA A2 and Apple M4:

```sh
python -m pip install -e '.[runtime,vision]'
DLLM_EXPERIMENTAL_VISION=1 DLLM_QUANTIZATION=q8_0 dllm serve \
  --model vision=/path/to/Qwen3.5-0.8B-snapshot --context 1024 --batch-size 4
```

Select the A2 UUID through `CUDA_VISIBLE_DEVICES` as for other CUDA runs.
For Apple, add `DLLM_DEVICE=mps` to the command above. Both FP16 and Q8 text
pass image correctness and lifecycle validation on M4 with Torch 2.13.0. A native Metal recurrent scan is used
for this qualified geometry at context <=1024 and at most four slots; set
`DLLM_HYBRID_METAL_SCAN=0` for the ordinary Torch diagnostic path.
`/v1/chat/completions` accepts user content lists with `text` and `image_url`
parts; image URLs must be inline PNG/JPEG data URLs with `detail: "auto"`.
The existing 1 MiB HTTP body limit applies. FP16 text is available by omitting
`DLLM_QUANTIZATION`; vision retains its separate FP16-storage/FP32-arithmetic
policy. Image tool calls, video/audio, remote image URLs, prefix snapshots,
and chunked image prefill are unsupported. Other model families keep their
existing text execution. See [image qualification](vision-qualification.md)
for independent output checks, lifecycle proof, and intermediate encoder limits.

For a complete local request, first start the opted-in server above with the
registered name `vision`. Build the JSON offline from a small still PNG or JPEG
(for example 256×256). This checks the encoded request size; the server still
validates decoded pixels, image tokens, prompt plus output context, and model
support. Renaming a file does not change its format.

```sh
python - photo.png > /tmp/dllm-image-request.json <<'PYIMAGE'
import base64
import json
import sys

with open(sys.argv[1], 'rb') as image:
    data = image.read(1024 * 1024 + 1)
if len(data) > 1024 * 1024:
    raise SystemExit('Image exceeds the request byte budget; use a smaller image.')
if data.startswith(b'\x89PNG\r\n\x1a\n'):
    mime = 'image/png'
elif data.startswith(b'\xff\xd8\xff'):
    mime = 'image/jpeg'
else:
    raise SystemExit('Expected a still PNG or JPEG image.')
request = {
    'model': 'vision', 'max_tokens': 32, 'temperature': 0, 'stream': False,
    'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this image briefly.'},
        {'type': 'image_url', 'image_url': {
            'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii'),
            'detail': 'auto'}},
    ]}],
}
body = json.dumps(request, separators=(',', ':')).encode('utf-8')
if len(body) > 1024 * 1024:
    raise SystemExit('Encoded JSON exceeds 1 MiB; use a smaller image.')
sys.stdout.buffer.write(body)
PYIMAGE
```

After that command succeeds, submit the request to your local server:

```sh
curl --fail-with-body --max-time 120 http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/dllm-image-request.json
```

The body contains the image itself; remove the temporary file when finished.
Change `stream` to `True` and add `curl -N` for SSE. A client timeout does not
prove successful server cleanup or model qualification. This example applies to
the qualified small Qwen image scope above; it does not extend it to practical
32K contexts or clear the recorded intermediate encoder error.

Gemma 3 text has a separate full-precision correctness path with alternating
sliding/full attention, GELU gating and its own normalization; the pinned 270M
checkpoint passes CUDA and Metal reference replay, including 511/512/513-token
window boundaries. CUDA uses FP16 weight storage with native FP32 matrix-vector
arithmetic and windowed decode attention; arithmetic and KV remain FP32.
`DLLM_GEMMA_WEIGHT_DTYPE=float32` selects full storage. Prefill optimization remains open. This does not imply support for multimodal Gemma. Qwen3.5 checkpoint qualification does not imply all newer Qwen
variants are supported. See the [model validation matrix](model-matrix.md):
Qwen3-1.7B and SmolLM2-360M pass the expanded tests; Qwen3-0.6B remains provisional
because two long-context positions exceed the logit-error limit.
Static linear and Llama 3 rotary scaling are now supported; dynamic/YaRN/LongRoPE
variants remain unsupported. The pinned Llama 3.2 1B checkpoint passes the A2
8192-context gate. See [architecture qualification](architecture-parity.md).
Chat requests also accept `stop` (one string or up to four strings, each 1–256
characters). Matching stop text is excluded from streamed and complete responses;
partial matches are withheld until they resolve. `max_completion_tokens` is an
alias for `max_tokens`; specify only one.

## Capacity and scheduling

Prompt plus requested output must fit `--context`. KV memory is reserved per
slot at that capacity. Fused attention reads the live prefix and skips unused
context blocks; larger configured contexts still add launch/merge overhead.

The server continuously batches up to four requests for one model by default;
use `--batch-size 1..8` to choose capacity. Different prompt lengths have separate
KV slots. Dense decoders and qualified short OLMoE batches pack available prompts
into shared linear layers; hybrid prompts currently prefill separately. Model groups drain
before switching; a waiting different model prevents later requests from jumping
ahead of it. Full prefill is the default, so a long prompt can delay active decode;
See [chunked prefill and prefix reuse](performance.md#chunked-prefill-and-prefix-reuse)
for qualified architectures.
Up to eight requests can be admitted, with a 120-second queue wait/backpressure
limit; HTTP connections are separately capped at 16.
Disconnects close the generation iterator once a socket write detects them.
The loopback development server has no authentication and is not a public gateway.

## Register, load and unload

You can start `dllm serve` with no models and manage its registry without a
restart. Run these commands on the server's host:

```sh
dllm add small Qwen/Qwen3.5-0.8B --revision 7ae557604adf67be50417f59c2c2f167def9a775
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

## Chat-template overrides

Use a per-model chat-template override when a checkpoint template needs replacing:

```sh
dllm add chat /path/to/model.gguf --chat-template /path/to/chat.jinja
dllm run /path/to/model.gguf --chat-template /path/to/chat.jinja --prompt "Hello"
dllm serve --model chat=/path/to/model.gguf --chat-template chat=/path/to/chat.jinja
```

The registration API accepts `chat_template` as literal UTF-8 template text (at
most 128 KiB), never a server-side file path. Templates use the same sandbox and
JSON serialization as checkpoint templates. Invalid syntax is rejected before
registration. Overrides remain attached to the alias across host/GPU eviction
and disk reload; status exposes the source and SHA-256. Registration identity
includes the path, quantization and template text. Startup overrides must name a
configured alias. Encoder, T5 and Mamba overrides are not supported. The default
always remains the checkpoint's embedded template unless explicitly overridden.
An override does not add an unsupported tool-call protocol to a model family.

The dllm HTTP extensions are `GET /v1/models/status` and
`POST /v1/models/register`, `/v1/models/load`, `/v1/models/unload`.
POST bodies use `model`; registration also requires `path`, and unloading accepts
boolean `release_host`. Commands share bounded admission and wait for active
model groups to finish. Ordinary generation still loads registered models on
demand. These extensions are separate from OpenAI's completion protocol.

## Structured JSON

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

## Sampling controls

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

## Token log probabilities

Chat requests can set `"logprobs": true` and `"top_logprobs": 5` (0–20).
Streaming and nonstream responses include selected-token scores and top
alternatives under `choices[].logprobs.content`, with token bytes where the
tokenizer provides a supported byte mapping. Scoring is opt-in and leaves the
default generation path without probability calculations. CUDA batches use
fused native scoring with one compact readback; Apple uses the qualified
PyTorch log-softmax fallback.

These are raw model log probabilities before sampling filters, bias, penalties
or grammar constraints. Custom text stops combined with scores, raw-completion
logprobs and prompt scoring are currently rejected. See the independent A2/M4
proof and measured overhead in [logprob evidence](evidence/logprobs/README.md).

## Function tools

Qwen2.5/Qwen3 JSON-call chat templates support `tools`, `tool_choice: "auto"`
or `"none"`, assistant `tool_calls`, and `tool` result messages with matching
`tool_call_id`. Streaming returns indexed tool calls after each complete call is
validated. dllm describes calls; your application executes them and returns results.

```json
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}],
  "tool_choice": "auto",
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 128
}
```

Auto mode does not enforce argument schemas. Other tool formats and
combining active tools with scores, custom stops or structured
response formats remain unsupported. Malformed/truncated calls stay text.
[A2/M4 tool-call and history evidence](evidence/tools/README.md).

Required/named tools use the optional structured-output dependencies. Set
`"tool_choice":"required"` to select one declared function, or
`"tool_choice":{"type":"function","function":{"name":"get_weather"}}`
to force one function. These modes constrain arguments to the supported bounded
JSON-schema subset and accept `strict:true`; explicit thinking is incompatible.
Automatic strict tool selection and multiple forced calls are not supported.


For the bounded two-rank network path, see
[resident distributed HTTP](multi-gpu.md#resident-distributed-http). Its greedy
sequential queue and single-model scope are narrower than this server.

Queued generation requests enforce the scheduler's queue timeout even while
another request is inside a long prefill. A timed-out request receives HTTP503,
is removed before execution, and returns its admission capacity. This deadline
covers only time waiting for worker ownership; an already-owned generation is
not interrupted when that queue deadline passes. Slow consumers retain the
separate stream backpressure timeout.

## Queue status during inference

Use `dllm status --server http://127.0.0.1:8088` or
`GET /v1/scheduler/status` to inspect queue ownership while inference is busy.
This reads scheduler bookkeeping without entering the inference queue or
inspecting GPU residency, so it also works when admission capacity is full.
`dllm models` remains the separate serialized model-residency view.

The response separates `queued_generation_requests` from
`worker_owned_generation_requests`, and `queued_commands` from
`worker_owned_commands`. Commands include model management, embeddings,
reranking and scoring. Worker ownership can include activation, prefill or
waiting for KV capacity; it is **not** a count of users actively decoding.
`queue_capacity` bounds all admitted requests, including owned work;
`batch_capacity` is the configured generation slot limit, not current occupancy.
`oldest_queued_seconds` is zero with no pending work. `closed` reports scheduler
shutdown. Counts omit completed/cancelled work and contain no prompt content,
tokens, model paths or request history. Runtimes without this scheduler return
HTTP501 instead of invented zero counts. This is operational visibility, not a
GPU-health check or sustained-user performance qualification.

## Finish admitted work before stopping

Run `dllm drain --server http://127.0.0.1:8088` to stop accepting new work while
already-admitted generations and commands finish in their existing order.
The equivalent request is `POST /v1/scheduler/drain` with JSON body `{}`.
It returns immediately and is idempotent. New generation and queued management,
embedding, reranking and scoring requests receive HTTP503. Existing queue and
backpressure deadlines still apply.

Poll `dllm status` until `drained` is true, then stop the process normally.
`draining` reports that admission is stopped; `worker_busy` stays true while the
worker owns an operation, even if its caller has cancelled. `drained` requires
both no admitted requests and an idle worker, rather than relying on an empty
client queue alone. Scheduler status, health, model listing and registry export
remain readable. Model-residency status uses the command queue and is rejected
during drain; use the scheduler view to follow progress.

Drain does not shut down the process, unload model weights or cancel an owned
GPU operation. It is one-way for this server instance; restart to admit new
work. Forced shutdown retains the existing cancellation behavior. Runtimes
without this scheduler return HTTP501 for drain.

Set serving admission limits explicitly when needed:

```sh
dllm serve --model chat=/path/to/pinned/snapshot --batch-size 4 \
  --queue-size 8 --queue-timeout 30
```

`--queue-size` counts all admitted work, including worker-owned requests, and
must be an integer at least as large as the batch size. It defaults to8.
`--queue-timeout` defaults to120 seconds and must be finite, positive and within
the platform thread-wait limit. It bounds waiting for worker ownership, not
inference execution. The same setting also bounds stream backpressure and
existing internal management waits; model-status commands retain their shorter
one-second queue bound. These options do not increase KV capacity or change the
HTTP connection cap. `dllm status` reports `queue_timeout_seconds` alongside
capacity. Invalid settings fail before downloads or GPU initialization.

## Remove a registered alias

Use `dllm remove ALIAS --server http://127.0.0.1:8088`, or
`POST /v1/models/remove` with `{"model":"ALIAS"}`, to release an alias's GPU and
host residency and remove its snapshot, precision and template registration.
The existing management barrier lets the current inference group finish before
removal; a management timeout prevents still-queued removal from executing later.
Requests behind the removal fail if they refer to the now-unknown alias.

Checkpoint files are never deleted, and other aliases remain registered even
when they use the same path. Missing aliases return an error. You can add the
name again with a corrected snapshot or precision. This differs from `unload`,
which preserves the registration. Save the live registry again with
`dllm models --save-config FILE` to persist the change; an old manifest still
contains the old alias. Exporting an entirely empty registry remains unsupported.

Successful batched generation timing includes `queue_seconds`: elapsed time
from admission bookkeeping to the worker taking the request. It excludes
activation, prompt preparation, KV-capacity waits after ownership, and decoding.
The nonstreaming response exposes it in `dllm_timing`; streaming responses with
`stream_options.include_usage:true` include the same `dllm_timing` extension in
the final usage event. Disabling usage does not add a separate timing event.


## Offline hybrid capacity planning

Inspect an existing local floating-point hybrid checkpoint without downloading
weights, reading tensor values, or initializing a GPU:

```sh
dllm plan-hybrid /path/to/local/snapshot --contexts 32768 65536 \
  --batches 1 4 --cache-dtype float32
```

The JSON report estimates native FP16, Q8 and Q4 weight storage using the real
text weight mapping, plus full-attention KV, FP32 recurrent state and convolution
history. State accounting is shared with runtime admission. A tied embedding
and packed output head are counted separately because they use different
representations. The default cache dtype is FP32; choose FP16 explicitly to
compare that storage scenario.

These are storage estimates, not runtime support, available-memory checks or
qualification results. The separately reported 512 MiB workspace admission
floor is not a peak workspace bound. Allocator overhead, graphs, communication,
host copies and loading peaks remain excluded. Local contents are not hashed,
so the report does not assert a model revision. Contexts must fit the model's
configured position limit and batches must be between 1 and 16.

## Intel XPU implementation preview

The Intel adapter is an **unqualified development path**, tracked in issue #63.
It routes synchronization, cache release, free-memory admission and residency
metrics through `torch.xpu`; it does not fall back to CUDA or CPU. The initial
eager executor accepts dense Qwen3 and Qwen3.5-family hybrid text floating-point
safetensors, uses FP32 device
weights/state, limits context to 2048 and batch capacity to four, and reserves
2 GiB of reported free device memory for the B580 desktop. This reserve is
additional to the executor's workspace reserve and does not guarantee desktop
responsiveness under every workload.

Development selection requires `--device xpu`, `DLLM_EXPERIMENTAL_XPU=1` and
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, with one visible Intel Arc B580 in an
XPU-enabled PyTorch environment. CUDA/Metal kernels, quantization, GGUF/AWQ,
multimodal models, scaled RoPE, paged KV, speculative decoding and graph
options are not part of this implementation. Ordinary decode graphs are disabled
for XPU. Local tests cover admission and API routing only: neither model
correctness, useful speed, real memory release nor the 32K product gate is
qualified. No Intel runtime package is added to the existing CUDA environment.

On the desktop, the installed oneAPI adapters initially failed to load because
`libumf.so.1` was absent from the SSH shell's library search path. Sourcing the
existing `/opt/intel/oneapi/setvars.sh` in that process resolved enumeration;
this did not require a driver update or persistent login-environment change.
SYCL enumeration is separate from having an XPU-enabled PyTorch installation.


The hybrid XPU path defaults to FP32 and uses the shared eager recurrence,
convolution and attention implementations. Its device budget counts promoted
FP32 weights rather than the smaller host weight representation. Chunked prefill
preserves convolution history and recurrent state across chunks, resetting only
at offset zero; later chunks attend to the existing KV prefix. A failed chunk
invalidates that slot's cursor so continuation is rejected until a new prompt.
CPU tensor regressions compare full prompts against several chunk boundaries,
subsequent decoding, interleaved slots and slot reuse. This verifies shared state
logic; it does not qualify Intel kernels or hardware memory behavior.


## Experimental host-weight layer streaming

`DLLM_EXPERIMENTAL_HYBRID_OFFLOAD=1` adds synchronous host-to-device layer
streaming to the CUDA hybrid text executor. It requires unquantized weights,
`DLLM_KERNELS=torch`, FP16 hybrid precision and `--no-graph`, with context at most
2048 and batch capacity at most four. This is an implementation preview under
issue #64, verified with CPU tensor and ownership tests only. No GPU execution,
transfer speed, practical-model quality or useful throughput is qualified.

The existing host cache retains the full source model within `--host-gib`.
Offload loading additionally requires Linux `MemAvailable` accounting and leaves
a 16 GiB host reserve. Before tensor loading, headers budget source storage plus
two FP32-sized copies per source element for conversion and concatenation. This
is deliberately conservative, including for tied weights. Source reads and CPU
conversion allocations recheck headroom. Missing or malformed accounting fails
closed; offline planning never queries it. These checks are observations, not
memory reservations: concurrent processes, allocator overhead and memory-control
group limits still require independent operational limits and monitoring.
Embedding, output head and final norm stay on the device; one layer at a time
is copied synchronously. Before replacing that layer, the store synchronizes
its device work, drops the old references and releases unused allocator cache.
Admission budgets the fixed weights, largest layer, all request state, workspace
reserve and the eager output head's temporary FP32 copy. Tied weights retain
shared storage. This first implementation has no pinned staging or prefetch,
and executes every layer on the GPU; it is not CPU layer computation.

Model status includes the active streamed layer, resident and peak owned weight
bytes, planned peak weight bytes, cumulative source bytes copied, transfer count
and failed/closed state. Copy counters are bookkeeping, not measured PCIe
traffic. A transfer failure poisons the store until eviction. Failed device
synchronization retains ownership rather than claiming buffers were released.
Closing the store releases its device and source references after successful
synchronization. Normal host-cache eviction remains responsible for releasing
the cached source model. Multimodal, packed quantization, sharding and graph
execution remain rejected for this preview.


The streaming path also participates in `--prefill-chunk-size` scheduling.
Every chunk preserves recurrent, convolution and KV state; the scheduler can
process cancellation and rotate admitted prompts between chunks. Layer transfers
within a chunk remain synchronous. CPU regressions cover the combined streamed
and chunked path, including an interleaved second slot. This adds scheduling
behavior, not a hardware latency guarantee.

`dllm plan-hybrid` also reports a `layer_streaming` storage scenario using the
same layer plan as execution: fixed weights, each layer's bytes, peak resident
weights and the output-head FP32 temporary. Its per-context rows add state and
the workspace floor. This scenario estimates unquantized source storage only;
it does not imply that a requested context, cache dtype, hardware or performance
level is admitted by the experimental execution policy. Existing fully resident
estimate fields retain their meaning.

The planner and safetensors loaders share checkpoint inventory validation.
Indexed shard paths must be safe relative paths; every indexed shard must exist,
and its tensor names must exactly match the index. Duplicate tensor names across
shards, duplicate JSON fields and malformed headers are rejected before tensor
values are loaded. Nested shards and Hugging Face cache symlinks are supported.
Inventory retains packed dtypes for their format-specific loaders; the hybrid
planner still accepts only floating-point source weights. Header validation does
not hash tensor contents or protect against concurrent changes to checkpoint files.

For explicit stage placement, add `--layer-splits 16 32` to `dllm plan-hybrid`.
The boundaries are global layer indices and must be strictly increasing inside
the model. Passing `--layer-splits` with no values reports one stage. Optional
`--stage-weight-dtype float32` estimates promoted FP32 weights; `native` retains
the text loader's FP16 matrices and FP32 vectors. Each stage row includes only
its weights and state, two bounded FP32 activation buffers, the final head's
conversion temporary where needed, and the workspace floor. These budgets use
the same calculation as `HybridLayerStage` admission. The planner does not
select devices, check their available memory or qualify the proposed split.

Instead of choosing boundaries manually, `--device-budgets-mib 22000 14000`
asks the offline planner to place contiguous, nonempty stages into those ordered
budgets. It minimizes the maximum estimated fraction of any budget consumed;
this balances storage, not compute or transfer time. Every requested context and
batch size gets its own result. A failed fit retains the requested limits and
reports `storage_plan_found: false`, without silently shortening context or
reducing users. Budgets are supplied inputs, not queried free-memory readings;
allocator fragmentation, loading peaks and hardware/transport qualification
remain outside the estimate. Planning is bounded to 16 stages and 256 layers.

The `build_cuda_pipeline(model, devices, budgets, max_batch=4)` helper
in `dllm.hybrid_pipeline_build` turns a loaded CPU model and ordered CUDA
ordinals/byte budgets into a synchronous coordinator. It requires
`DLLM_EXPERIMENTAL_HYBRID_PIPELINE=1`, CUDA, unquantized FP16 text, context at
most 2048 and at most four slots. It rejects offload, legacy layer splitting,
vision and paged KV.
It plans before allocating, bounds the supplied caps by observed free memory,
then rechecks each device before constructing its stage. Storage fit does not
establish practical execution speed or numerical qualification on any GPU.

Construction failures raise `PipelineBuildError`. The builder synchronizes all
devices it touched before releasing completed or partial stages. If cleanup
also fails, the exception retains ownership: keep it and retry its `close()`
after recovery; its `closed` flag records completed cleanup. A successfully
constructed coordinator owns its stages and must likewise be closed. CPU
fixtures cover arithmetic equivalence, changed headroom, partial allocation,
interruption and failed-cleanup retry; device transport remains unqualified.

For experimental serving, set `DLLM_HYBRID_PIPELINE_BUDGETS_MIB` to comma-separated
positive integer limits, one per visible GPU in `CUDA_VISIBLE_DEVICES` order,
alongside the pipeline opt-in. For example, `22000,14000` specifies caps of
22000 and 14000 MiB; the UUID selection must expose exactly two distinct GPUs.
Use `--device cuda --no-graph`, unquantized hybrid text and the bounded context
and batch limits above. The ordinary runtime then selects the pipeline for
chat/completions and streaming; `--prefill-chunk-size` remains available. Prefix
snapshots are rejected. Each stage executes eagerly and transfers hidden states
synchronously; no overlap, transport speed or GPU serving qualification is claimed.

Status reports each resident pipeline's stage storage and total CUDA allocations
across its visible devices. Model unload/swap closes all stages. If build cleanup
fails, `pending_pipeline_cleanup` remains nonzero and subsequent activation must
complete that cleanup before loading another executor; unload also retries it.
CPU/loopback regressions exercise the actual stage arithmetic through the HTTP
handler and scheduler, including concurrent requests and host-cache restoration.

Scheduler shutdown cancels queued work, waits for the worker to exit, then closes
the runtime under its request-lifetime lock. Runtime close rejects new loads and
requests, retries pending build cleanup, closes all resident executors and drops
host-cache references. If the worker or device cannot finish, shutdown raises an
error and retains the remaining ownership for a later `close()` retry. It never
frees a model while its worker is still active. Successful close is idempotent.
Failed model rebinding also retains its executor outside the serving registry
until cleanup succeeds. `pending_executor_cleanup` reports these owners; new
activation, unload and shutdown retry their cleanup before releasing host state
or loading replacement executors. The original rebind error retains a cleanup
failure note, and a partially rebound model is never published under an alias.

## Optional API authentication

Set `DLLM_API_KEY` for inference clients and a different `DLLM_ADMIN_API_KEY`
for administration before starting the server. Each configured key must contain
16–512 printable ASCII characters without spaces; use independently generated
random values. Keys are read at startup and never included in status, manifests
or error responses. Restart to change them. If neither variable is present,
and no named clients are configured, the existing anonymous loopback behavior remains. An explicitly empty or invalid
key is an error, not a request to disable authentication.

Clients send `Authorization: Bearer <key>`. The inference key permits model
listing, chat/completions, embeddings, reranking and token scoring. The admin key
also permits registry/residency operations, configuration export, scheduler
status and drain. Configuring only an inference key leaves administration
inaccessible through that key; an admin-only configuration allows the admin key
to perform both roles. `/health` stays credential-free. Invalid credentials get
401, and inference credentials used for administration get 403. Authentication
precedes body reads and runtime admission, and duplicate authorization headers
are rejected.

Management CLI commands read the admin key, falling back to the inference key
when only that one is configured (administrative requests then receive 403).
`dllm bench` continues to use `DLLM_API_KEY`. Authenticated CLI and benchmark
requests refuse redirects; use the endpoint's final URL explicitly. Anonymous
client behavior is unchanged. These roles do not provide TLS or tenant data
isolation. The server still binds only to
`127.0.0.1`; no public listener or deployment is enabled.

For separately limited inference clients, set `DLLM_CLIENT_KEYS` to a JSON object
mapping client names to `{"key":"<independently-generated-secret>","max_inflight":2}`.
Names contain 1–64 letters, digits, dots, underscores or hyphens. Configure at most
128 clients, each with a distinct key meeting the rules above and an integer
`max_inflight` from 1 to 128. Duplicate names/fields, duplicate keys, unknown fields,
empty objects and invalid limits fail startup. Keys cannot duplicate the shared
inference or admin credential. Restart to change client configuration.

A named client has inference access only. Its limit counts outstanding inference
HTTP requests, including body reads, queued work and streaming responses. Excess
requests receive `429` with `Retry-After: 1` before body reads or scheduler admission.
The slot is released when the handler finishes, including failure/disconnect
cleanup. Limits are shared across that server's handler threads and independent
between clients; they do not replace global connection or scheduler queue limits.
Shared inference/admin keys retain their existing unrestricted per-client behavior,
so omit the shared inference key when every inference client must have a cap.
Named clients may add `"models":["alias-a","alias-b"]` to restrict inference to
exact registered aliases and filter `/v1/models` discovery. Omit `models` to retain
access to all models; an empty list denies every model. Lists contain at most 128
distinct aliases and use the same alias validation as model registration. There
are no wildcards or path normalization. Forbidden models receive a generic `403`
before runtime dispatch on all inference routes, including streams, embeddings,
reranking and token scoring. Model aliases are authorization identities: an
administrator changing an alias's checkpoint also changes what that permission
allows. Shared inference/admin keys retain access to all aliases.

When prefix snapshots are enabled, each named client has a separate opaque cache
scope. Exact and partial reuse require the same client scope, including cached
logits; the scope follows queued requests into both scheduler prefill paths.
Request payloads cannot select a scope. Shared inference/admin credentials and
anonymous requests use one shared scope per server handler policy. Direct local
runtime callers retain their separate default shared scope. Global cache token and
entry limits still apply, so clients can still evict each other's entries; this does
not reserve per-client capacity or provide process/data isolation, token-rate
fairness or distributed quotas. Client-only configuration still protects the API;
`/health` stays public.

Named clients may set `"max_reserved_output_tokens":4096` to cap the sum of
declared output limits for queued/running generations. The value must be an
integer from 1 through 2147483647; omission leaves this budget unlimited.
Both completion routes reserve their `max_tokens` limit (default 128); chat
also accepts `max_completion_tokens`. Reservations begin after request validation,
before runtime dispatch, and end when the generation iterator closes, including
failure or cancellation. Excess commitments return 429 with `Retry-After: 1`.
This conservative budget is separate from request-count limits and does not
measure actual token rate, prompt/KV memory, or quotas across server processes.
Shared/admin credentials and non-generation routes retain their existing policy.

Named clients may set `"max_prefix_cache_tokens":4096` to limit retained prefix
tokens in each executor's cache. Omission preserves global-only limits; zero
disables snapshot retention for that client. Values are integers from 0 through
2147483647. When adding a snapshot, the cache evicts that client's least recently
used snapshots to fit its cap before applying the global token/entry limits.
Rewindable snapshots may retain a shortened prefix; recurrent snapshots that
cannot rewind are skipped when the full prefix exceeds the cap. The cap follows
the opaque request scope through both scheduler paths. It limits retained token
counts, not bytes or active-request KV memory, and does not guarantee reserved
capacity or aggregate quotas across executors/processes.

Named clients may set `"max_requests_per_minute":120` for a rolling 60-second
inference admission limit. Values are integers from 1 through 60000; omission
is unlimited and explicit null is invalid. The limit spans all inference routes
and is checked atomically with the outstanding-request limit before body reads
or runtime dispatch. Rejected rate/outstanding admissions do not consume quota.
Accepted attempts do consume quota even if payload validation or model execution
later fails, or the client disconnects; finishing a request does not refund it.
Rate rejection returns 429 with a whole-second `Retry-After` rounded up to the
oldest counted attempt's expiry. Other clients, shared/admin keys and discovery
or management routes keep their existing policies.

This uses a monotonic rolling window shared by handler threads in one server.
History is bounded by each configured request cap, pruned on the next attempt,
and resets on process restart. It is request admission control, not actual token
rate fairness, a persistent/distributed quota or a per-client capacity reservation.

### Client turns within generation groups

Named clients use their existing opaque request scopes for scheduler admission
turns. Within the leading contiguous generation group for one model, the client
least recently admitted receives the next slot; FIFO is preserved within each
client. A newly waiting client can receive a turn ahead of another client's
queued burst. Shared/admin/anonymous generation remains one lane. Identity state
is bounded by waiting lanes and cleared when idle, on a model change, or at a
management command. Cancelled/expired entries are removed before selection.

Turns never cross a different-model or management-command barrier, preserving
model draining and command order. Embedding/reranking/scoring commands remain
serialized barriers. This is request admission fairness, not token-weighted
service, distributed quotas or a measured user latency/throughput SLO. Active
requests retain their existing execution and cancellation behavior.

Named clients may also set `"max_output_tokens_per_minute":8192` to limit
**declared output admission** across completed and active generation requests.
Values are integers from 1 through 2147483647; omission is unlimited and explicit
null is invalid. Both completion routes charge `max_tokens` (default 128), or
chat's `max_completion_tokens` alias, after request validation and before runtime
dispatch. A request larger than the entire allowance receives 400. Other output
budget rejections do not charge this allowance. The independent request-count
limit still counts admitted HTTP attempts according to its own policy.

Charge and outstanding-output reservation checks are atomic. Accepted output
charges persist after early EOS, execution failure, cancellation or disconnect;
unused declared tokens are not refunded. Output rate rejection returns 429 with
a calculated whole-second `Retry-After` until enough weighted budget expires,
not merely the oldest request. Each client retains at most 61 one-second buckets.
Charges expire 60 to 61 seconds after admission, conservatively rounded to the
end of their admission second; this never permits a larger 60-second burst.

This is a process-local admission allowance, not measured token throughput,
input-token accounting, weighted execution fairness or distributed billing.
Restarting the process resets it. Other clients, shared/admin credentials and
non-generation routes retain their existing policies.

### Optional client decode shares

`dllm serve --client-decode-turns` gives each currently ready client an equal
number of rows in each ordinary decode step by default. For example, with three ready
requests from client A and one from B, the step decodes one A request and the B
request; A's requests rotate on later steps. With four A requests and two B
requests, each gets two rows. When only one client is ready, all its ready rows
can decode. The default remains disabled and decodes every ready request.

Named entries in `DLLM_CLIENT_KEYS` may set `decode_weight` to an integer from
1 to 16 (default 1). With decode turns enabled, ready token rows are proportional
to these weights across iterations. For example, two continuously ready clients
with one request each and weights 3 and 1 decode the first request every step and
the second every third step. With enough ready rows, a step can serve multiple
rows from each client; requests within a client rotate. One ready client still
uses its full batch. With the option disabled, weights have no scheduling effect.

Weights are administrator policy, captured in immutable opaque scopes; clients
cannot choose weights in request bodies. Fractional credits are exact bounded
integers smaller than one token row, retained only for currently ready lanes.
An absent or fully backpressured lane loses its credit and earns none while away.
Weights count rows, not request admissions, prefill work, compute time or wall
clock service. The policy makes no throughput or latency guarantee.

The policy uses the opaque named-client identity captured before scheduler
handoff. Anonymous/shared-key requests form one lane; request payloads cannot
select another client's lane. Backpressured requests are omitted without blocking
other clients or accumulating future service credit. Rotation state retains at
most the current ready batch and resets at model-group boundaries. Cancellation,
page release, model/management barriers and sampling retain their existing rules.
Ordinary, chunked and paged scheduler paths share this selection policy.

`GET /v1/scheduler/status` reports the boolean `client_decode_turns` setting. This
option trades potential batch utilization for configured shares across ready clients;
no throughput improvement is claimed. Shares count decode rows, not computation
time or wall-clock token rate. Prompt work and the first token from prefill are
outside this policy. It does not reserve slots for waiting clients, provide
weighted quotas, isolate prefill cost, or establish a latency/throughput SLO.
Local scheduler controls preserve per-request output and release behavior; real
hardware measurements are still needed to quantify the cost.

### Optional client prompt-token turns

`--client-prefill-turns` requires an explicit positive `--prefill-chunk-size`
(at most 1,048,576 tokens) and a chunk-capable executor. It chooses the pending
client with the least weighted prompt-token service, then rotates that client's pending
requests. A client occupying three prompt slots therefore does not receive three
times the aggregate token service of a client with one slot. A short final chunk
is charged for its actual token count. Default per-request rotation is unchanged.

Accounting includes only dispatched prompt chunks, excluding reused prefix tokens.
New/rejoining clients start at the current service floor; cancelled/completed
requests and absent lanes lose history. Normalized service counters remain within
one maximum dispatched chunk across continuously pending lanes, and all state
clears when prompt work/model ownership ends. Decode still runs between chunks.
Named `DLLM_CLIENT_KEYS` entries may set `prefill_weight` to an integer from 1
through 16 (default 1), independently of `decode_weight`. Each dispatched chunk
charges its actual token count divided by this declared weight. Continuously
pending clients with weights 3 and 1 therefore approach a 3:1 prompt-token share,
within one chunk of normalized service. Exact integer accounting uses a common
unit divisible by every allowed weight; counters are rebased to the live minimum
and bounded by the largest normalized chunk. Equal weights preserve the existing
selection order, and anonymous/shared-key clients use weight 1.

Weights are administrator configuration copied into immutable opaque scopes
before handoff. Request bodies cannot choose a scope or weight. Without
`--client-prefill-turns`, prefill weights have no scheduling effect. Decode
weights remain independent; neither weight changes request/rate/output quotas.

This applies only to admitted, runnable prompt work. It does not reserve slots
for queued clients, preempt a running chunk, equalize attention/computation cost,
coordinate quotas across processes or guarantee proportional wall-clock service. It establishes
no wall-clock latency or throughput SLO. Scheduler status reports the boolean
`client_prefill_turns` setting.

### Shared local-process rate accounting

Set `DLLM_CLIENT_RATE_DB` to an explicit absolute SQLite file path to share named
client rolling request and declared-output allowances across server processes on
one host and retain them across restart. Without this variable, existing
in-memory behavior is unchanged and no database is created. Configure at least
one named client's rate limit before opting in.

Use a dedicated local directory owned by the server user and not writable by
other users/groups. The file is created with mode 0600; existing files must be
private, regular, single-link files owned by that user. Symlinks, FIFOs, unsafe
permissions, replaced files and incompatible/corrupt databases fail closed.
Network filesystems and cross-host sharing are unsupported.

All processes sharing a named client must use identical request/output rate
limits. Other client settings may differ, and rotating a key while keeping the
same name and rate policy preserves its counters. The database stores only client
names, rate limits, a clock high-water mark and count buckets—no API keys,
prompts or output text. It holds at most 128 registered names, 61 live buckets
per client and 32 MiB of database pages. Names are not silently evicted to make
room for new policies.

Transactions atomically charge admitted attempts and declared output tokens.
Both dimensions use conservative second buckets expiring after 60–61 seconds;
this differs from the default in-memory request counter's exact 60-second window.
`Retry-After` waits for enough budget, including after a backwards wall-clock
change. A persisted per-client high-water mark prevents clock rollback from
reopening previously consumed allowance. Forward system-clock changes still
advance expiry; this is local accounting, not a trusted billing clock.

Rejected rate charges are not added; admitted validation failures, cancellation,
disconnection or model failure are not refunded. Output above the entire allowance
still fails request validation. Shared/admin keys retain their unlimited rate
policy. Without the separate shared-reservation option, outstanding-request/output
reservations and cache limits remain process-local; this option does not create persistent inflight leases or a
cluster quota service.

Lock waiting is bounded to one second per SQLite connection. Missing/replaced,
locked-too-long or corrupt storage returns 503 before model dispatch rather than
falling back to a fresh allowance. Process-local reservations release on these
failures. Filesystem I/O itself is not preempted by that lock timeout. Stop all
users before deliberately replacing/resetting the database or changing a stored
client's rate limits; running owners reject inode replacement. No automatic
reset, migration, credential storage or deployment is performed.

### Shared outstanding admission across local servers

Set `DLLM_CLIENT_RESERVATION_DB` to the same absolute private local SQLite path
for participating named-client servers. This coordinates `max_inflight` and
`max_reserved_output_tokens` independently of rolling rate limits. Defaults remain
process-local. Owner locks permit crash reclamation without unsafe lease expiry;
uncertain release remains charged and requires cleanup retry or a controlled
restart. See [storage bounds, shutdown and recovery](shared-client-reservations.md).
This does not coordinate different machines or reserve GPU memory.

Explicit embedders can also pass a [remote quota admission adapter](remote-quota-admission.md)
to the handler. It verifies authority policy and retains uncertain operations
through the same HTTP cleanup path. Network-backed admission is not enabled by
default, and prior unresolved remote claims block owner startup.

Explicit cross-host outstanding quotas are configured with `DLLM_REMOTE_QUOTA_CONFIG`; see [quota commands](quota-commands.md) for authority bootstrap, pinned mutual TLS, owner recovery and CPU command shutdown behavior.

Serving commands own accepted HTTP request threads through shutdown. SIGINT or
SIGTERM stops ordinary server admission, then closes the scheduler before joining
request threads and releasing their quota leases. Listener/configuration failures
also close the owned runtime. A runtime cleanup error does not skip HTTP cleanup;
failed owners remain the caller's responsibility and the command does not report
success. Incomplete request bodies can hold shutdown until the existing socket
read timeout. This is cooperative cleanup, not a kernel execution deadline.
`BoundedHTTPServer` itself keeps its existing embedder defaults; custom owners
must choose and manage their own request-thread lifecycle.

Structured schemas must contain valid Unicode throughout their values and keys. Lone surrogates hidden by JSON escapes are rejected during request validation, before grammar compilation; the existing ASCII-serialized 16 KiB schema budget is unchanged.

Integer literals in schema `const` and `enum` constraints must fit signed 64-bit range, including literals under `type: number`. Larger literals are rejected because the grammar compiler can round them through floating point. Unconstrained integer schemas and finite floating-point constants remain supported.

Native ByteLevel, byte-fallback and metaspace logprob token text and bytes follow the supported decoder mapping for ordinary added tokens, including leading-space glyphs and partial UTF-8 pieces. GGUF user-defined literal pieces retain their separate literal-byte contract; special controls remain hidden.

Generic function-call arguments must decode to valid UTF-8 JSON, including nested keys and values. Escaped lone surrogates are not published as calls or accepted in tool history. Automatic malformed envelopes remain plain text; constrained calls retain their existing invalid-output and length-termination behavior. Valid Unicode and escaped surrogate pairs remain supported.

Tool definitions also require valid UTF-8 in descriptions and nested schema keys/values before generation dispatch. The existing 256 KiB ASCII-serialized definition limit is unchanged; escaped invalid Unicode does not bypass validation.

Remote quota RPC deadlines include response decoding and receipt validation. A response that finishes late is an unknown outcome even if the authority committed it. The reservation remains charged; recovery must retry the identical immutable request to obtain its durable receipt without a duplicate charge.

Embedding text and reranking queries/documents must be valid UTF-8. Lone
Unicode surrogates supplied through JSON escapes return HTTP 400 before a
model session is acquired. Valid Unicode text and explicit embedding token IDs
retain their existing behavior.
