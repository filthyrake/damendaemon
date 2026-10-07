# Native GLM generation on a prepared CPU host session

`dllm.glm_host_generation.HostGlmGenerator` explicitly binds the existing native
GLM reasoning/text and schema-bound tool format to a ready GLM host coordinator.
It does not admit the model to generic HTTP serving or change the raw-token
command. The peer owners must continuously step their workers.

Every stage's `HostTopology` must use `fp8_reference`, the native tokenizer's
mapped vocabulary size, no extra mapped-token exclusions, and
`native_session_identity(text, verified=...)` as its session identity. The
source snapshot identity remains separately bound by the topology. The native
identity binds the pinned renderer, canonical tokenizer, padded and mapped
vocabulary sizes, native stops, and source-verification mode. The adapter
snapshots a `GlmToolText`; subsequent mutation of the caller's tokenizer does
not change the admitted policy. Construction requires a ready, empty,
unclaimed coordinator.

`generate` and `generate_tools` accept native message history and bounded output
length. `generate_batch` and `generate_batch_tools` accept up to the negotiated
slot count, per-row output limits, and optional `ragged=True`. Invalid histories,
tools, context limits, batch shape, and policy fail before model dispatch. Tool
calls are parsed and reported; no tool is executed. Text/analysis channels,
row identities, finish reasons, and prompt/completion token counts match the
local native generator. Tool-format tests select controlled tokens after actual
stage execution; they do not establish trained-model tool competence.

Success finals follow all acknowledged slot releases and parser cleanup.
Healthy completion permits reuse. Active cancellation, iterator abandonment,
remote failure, or parser cleanup failure discards the epoch. Retain the owner
and retry `cleanup()` or `close()` when cleanup raises; create a new session to
serve more requests after epoch discard. `close()` can interrupt blocked network
work from another thread. Streaming pieces already delivered cannot be recalled.

Tests use generated packed checkpoints over actual local TLS, compare native
text/usage against independently expanded whole-model controls, and exercise
ragged tools, policy rejection, snapshot isolation, cancellation, peer failure,
and retained cleanup. This is CPU reference evidence only; official checkpoints,
GPU execution, practical context/capacity, and network performance remain open.
