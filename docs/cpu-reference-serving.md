# Modern CPU reference chat HTTP

`damendaemon.reference_serving.ReferenceRuntime` explicitly owns one prepared
`GlmToolReferenceGenerator`, `FlashToolReferenceGenerator` or DeepSeek
`ReferenceGenerator`. Use `handler_for_reference` with the existing bounded HTTP
server to expose it on a caller-chosen loopback listener. This adapter does not
load weights, download models, select a GPU or broaden production model admission.

The ordinary chat endpoint, authentication, per-client admission and SSE handler
are reused. Native tool support is an explicit runtime capability: these runners
receive their family-specific tool declarations and return parsed structured calls;
the legacy JSON-envelope parser does not reinterpret GLM or DSML wire text. Existing
production runtimes keep their previous parser and strict-tool-choice behavior.

Supported requests:

- `/v1/chat/completions`, text conversations and validated tool/result history.
- Default greedy sampling and 1–4096 output tokens within the runner's context.
- Native `tool_choice: auto` or `none`; required/named forcing is rejected.
- GLM/Flash `reasoning_effort: low` or `high`, with the pinned default when omitted.
- DeepSeek boolean `chat_template_kwargs.enable_thinking`, default false.
- Reasoning/content channels, atomic native calls, streaming and final usage.

GLM/Flash use the bounded schema-bound tool wire contract even if `strict` is
omitted. DeepSeek enforces parameter schemas for explicit `strict: true`, retaining
its non-strict compatibility behavior. The shared HTTP input limits and identifier
rules still apply. Generated call IDs are fresh for each response, so carrying a
second tool turn into history does not reuse the previous call's identity. Tools
are returned as data and never executed.

Raw completions, media, non-default sampling, custom stops, logprobs, constrained
response schemas, model changes, embeddings, reranking, scoring and production
model-manifest export are rejected. No unsupported option silently falls back to
a different inference policy. Invalid tools/history fail before model prefill.

Only one iterator may execute at a time. Another request receives 503 instead of
sharing mutable runner state or entering an unbounded queue. Close abandoned
iterators; their owned model caches are released. Cleanup failure remains reachable
and blocks later prefill until retry succeeds. Status reports pending cleanup;
health returns 503 while cleanup is pending or shutdown has begun.

`runtime.close()` prevents new work and asks an active iterator to stop at its next
event boundary. It raises BusyError while that iterator still owns execution;
retain the runtime, close/wait for the iterator, then retry. It cannot interrupt a
CPU prefill in the middle of a kernel. Successful cleanup drops the runtime's runner
reference; callers must not keep using a runner transferred into this owner.

Final responses follow successful cache cleanup. Timing explicitly reports
`source_tier: cpu_reference`, a family-specific CPU reference backend and
`official_model_qualified: false`. Total time includes HTTP consumer pacing and is
not isolated engine performance. The local gate covers scripted native formats,
real packed GLM arithmetic through HTTP, legacy handler regressions and lifecycle
failures. Official-weight quality, accelerator execution, practical long context,
continuous batching and multi-user SLO qualification remain open under #53/#56–#59.

## Explicit local command

Run a prepared local snapshot without registering it as a production GPU model:

```sh
python -m scripts.serve_cpu_reference glm /path/to/snapshot --alias local-glm --context 4096 --port 8088
```

Select `glm`, `flash` or `deepseek` explicitly. The command checks the selected
family, tokenizer/configuration and the existing `DAMENDAEMON_API_KEY`,
`DAMENDAEMON_ADMIN_API_KEY` and `DAMENDAEMON_CLIENT_KEYS` policy before loading tensor payloads.
It does not download weights. Authentication is optional under the existing
loopback policy; `/health` remains public. With a key configured, model listing and
inference require it. No credentials are printed.

The listener is fixed to `127.0.0.1`; port zero selects an ephemeral port. A JSON
`http_ready` line provides its URL only after model loading and a successful bind.
`--connections` caps connection threads (default 16, maximum 64); execution still
allows only one active request and returns 503 for another.

All families use FP32 CPU arithmetic. `--weight-mib` caps retained model weights,
`--cache-mib` caps request caches, and `--projection-workspace-mib` caps projection
scratch. Defaults are 256, 256 and 8 MiB. GLM/Flash additionally accept
`--workspace-mib` and `--output-mib` (both default 64 MiB); DeepSeek rejects those
options because its loader has no corresponding independent budgets. Component
budgets do not constitute a total process-memory cap. Defaults are deliberately
small reference limits, not recommendations for official model deployment.

SIGINT/SIGTERM stops new admission and requests cooperative cancellation. The
command joins HTTP request threads, retries owned cache cleanup, then emits
`stopped`. A CPU kernel cannot be interrupted mid-operation; shutdown can wait for
it and for an existing socket's 30-second timeout. Failed cleanup exits with an
error instead of reporting successful shutdown. DeepSeek failed-prefill caches
are included in the runner's cleanup and the HTTP pending-cleanup status.

Tiny packed local snapshots exercise each family's command through real HTTP/SSE
and process shutdown. This verifies wiring and lifecycle, not useful official-model
answers or GPU performance.

## Flash still-image HTTP opt-in

```sh
python -m scripts.serve_cpu_reference flash /path/to/snapshot --flash-images --context 4096 --port 8088
```

This selects a distinct prepared `FlashImageToolReferenceGenerator`. Each request
must retain at least one inline PNG/JPEG `image_url` part in a user message. Send
text and images as ordinary typed content parts; tool declarations and returned
assistant/tool-result history use the same native schema-bound contract above.
Tools are data only. Text-only requests use the default command without the image
flag. No remote URLs, filesystem paths, video, audio or animated images are loaded.

The existing HTTP body limit stays 1 MiB. At most four images and 16 million total
decoded pixels are admitted; image processing has its own patch and byte budgets.
Family history and encoded framing are checked before pixel decoding; the runtime
acquires its single request owner before decoding pixels or running vision. Image
context/patch/scratch failures occur before vision/text execution. A busy owner
returns 503 without decoding the competing request's pixels.

Image startup validates the pinned processor class/geometry and image-token
identity before loading payloads. `--vision-weight-mib` defaults to 256,
`--image-workspace-mib` and `--image-output-mib` to 64, and `--max-patches` to 4096.
These options require `--flash-images`; other families reject the flag.
`--workspace-mib`/`--output-mib` also bound vision execution scratch/output; they
are distinct from image preprocessing budgets. Readiness and model status report
`modality: still_image`, with the same CPU-reference qualification boundary.

The local gate covers actual tiny packed text+vision snapshots through a separate
HTTP process, native tool/SSE/history routing, malformed/remote/over-budget input,
concurrent rejection and cache cleanup. It does not establish official-checkpoint
image semantics, accelerator support or practical-model performance.

## Optional bounded serialized admission

`serve_cpu_reference` accepts `--queue-size 1..64` and `--queue-timeout` within
300 seconds. The default size is zero, preserving immediate busy rejection.
`QueuedReferenceRuntime` exposes the same opt-in to prepared Python callers.
Text, native tools and Flash still images retain their existing contracts.

One owned worker performs generation and cache cleanup. Named clients receive
the existing opaque-identity admission turns, preserving FIFO within a client;
HTTP authentication, model permissions and outstanding/output quotas remain in
force while waiting. This is serialized service, not continuous batching or
simultaneous execution. Images are decoded only once their request owns execution.

Requests snapshot at most 1 MiB of JSON metadata at admission; later caller
mutation cannot change them. Queue expiry is checked before prefill. Full queues
and expired requests return 503. Each stream buffers at most eight data events
plus one final/error event. A slow consumer backpressures the one owner, so other
requests can reach their queue deadline; buffering does not reserve a latency
SLO or bound total process memory. Cancellation closes the current iterator and
releases its cache before another request executes. Pending cleanup failures
stop admission and retain the owner for explicit cleanup retry.

`/v1/scheduler/status` stays responsive while generation is busy; the existing
admin-authorized drain route stops new admission and lets accepted work finish.
Drain does not exit the service. Shutdown wakes queued clients, cooperatively
stops active generation at a token boundary, joins the worker (up to five
seconds per close attempt), and closes the prepared reference owner. A busy
close retains ownership for retry; this is not preemption of a CPU kernel or a
hard generation deadline. Queue timing is reported separately from execution,
and all metadata remains explicitly CPU/reference/unqualified.


CPU reference cancellation is checked before prefill, between generated tokens
and before further decode, including while a native tool parser buffers a call.
Runtime shutdown and queued-request abandonment are combined in an execution
scope that is reset before the next request and does not cross threads. Running
kernels and image encoding are not preempted; cancellation is cooperative after
control returns. Cancelled calls produce neither a tool result nor a success
final, and caches/parser buffers are closed through the existing cleanup path.
DeepSeek single and batch generators now explicitly close their parser buffers
on abandonment as well as normal completion. Failed cache release remains
owned and retryable, preserving the cancellation error as the primary cause.
