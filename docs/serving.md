# Serving and model management

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Use one server for registered model aliases, streaming requests, and explicit residency control.

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
