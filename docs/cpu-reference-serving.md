# Modern CPU reference chat HTTP

`dllm.reference_serving.ReferenceRuntime` explicitly owns one prepared
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
