# GLM CPU host topology completion

`python -m scripts.run_glm_host_topology` is the explicit GLM counterpart to the
Qwen CPU topology command. It accepts `worker`, `complete`, and explicit native `serve` roles, an
admitted `glm_moe_dsa` snapshot and `fp8_reference` precision. It never selects an
accelerator, discovers hosts or provisions services. Start workers yourself.

Use the same bounded plan and ordered literal-IP peer files documented in
[cpu-host-topology-command.md](cpu-host-topology-command.md). Every cut must
preserve full/shared indexer groups. For a four-layer fixture, an example plan is
`{"ranges":[[0,2],[2,3],[3,4]]}`; actual checkpoints need their own valid cuts.
Workers receive their nonzero `--index`, bind address/port, local CA/certificate/
key and pinned coordinator certificate. Index zero receives the ordered peers
file and client TLS credentials. All participants need identical plan, model ID,
context/slot/frame/chunk limits and tokenizer/EOS policy.

The default model ID is an operator assertion. With `--verify-model-id`, every
process hashes the bounded snapshot before opening its endpoint/connection and
checks the receipt again after loading. The identity binds verification policy,
tokenizer serialization and EOS in addition to topology/configuration. Only the
assigned stage weights are decoded and retained; optional verification still
reads the whole selected snapshot. Worker weight/cache/workspace allowances are
per process, not a global cluster budget.

By default, completion accepts raw `--prompt`, a `--tokens` file or a bounded
`--batch-inputs` file. Opt into `--native-chat` on **every** process for the pinned
GLM native text/reasoning/tool format. The coordinator then accepts `--prompt`,
`--messages history.json`, or `--batch-messages conversations.json`. A batch file
is a JSON list of conversations, each an ordinary native message list. Use
`--ragged-batch` for unequal encoded prompt lengths, `--tools tools.json` for
function declarations, `--reasoning-effort low|high|max`, and optionally
`--clear-history-reasoning`. Workers receive only `--native-chat`, not request
options. Tools are returned as calls and never executed.

Native mode validates the entire request against the negotiated slots, context,
output and activation frame budget before connecting or loading payloads. Input
JSON files are bounded to 1 MiB. Native and raw session policies cannot mix;
all stages bind the pinned renderer, canonical tokenizer/stops and verification
mode. Native mode emits semantic content/reasoning/tool events and per-row usage,
not raw token events. Its success finals still wait for every STOP acknowledgement.
The `ready` event identifies `native_chat`. Arbitrary templates remain unsupported. GLM projections retain the supported packed FP8 source values with
CPU FP32 reference arithmetic; there is no alternative quantization mode here.

Events identify `glm_cpu_host_topology_reference`, all ranges and whether this
process verified source bytes. Completion success finals are withheld until
every worker acknowledges STOP. Setup/peer failure and cooperative signal cleanup
close the owned session without emitting a successful completion. This is not a
hard deadline for interrupted CPU kernels. Tests run three/four local processes
with ephemeral mutual TLS credentials and compare raw tokens, text and usage with
independent floating controls; they cover source/verification/EOS disagreement,
signalled listeners, partial connection failure and pre-network rejection.
Native subprocess tests also compare single/ragged text and tools with independent
whole-model controls, and reject malformed inputs, raw/native disagreement and
failed STOP without success finals. Tool-wire competence is covered separately
with controlled selection after actual stage execution; random fixture outputs
are not evidence of trained-model tool quality.
Physical networks, official model behavior, practical context and performance
remain unqualified.


## Native loopback HTTP

Use `serve --native-chat` at index zero with the same plan, peer file, native
workers, TLS files and identity policy. `--http-port` defaults to 8088 (0 chooses
an available port), and `--alias` names the served model. The existing
`DLLM_API_KEY`, `DLLM_ADMIN_API_KEY` and named-client policies apply; secrets stay
in the environment. HTTP binds only to loopback. Readiness and model status
identify CPU reference scope, topology and source-verification state.

This path owns one HTTP generation at a time. Healthy JSON/SSE native chat/tool
requests release slots for reuse. `--queue-size 1..64` explicitly adds bounded
serialized admission; `--queue-timeout` bounds waiting time (up to 300 seconds).
The default queue size is zero and concurrent requests receive busy responses.
Queued admission preserves per-client turns and bounded event buffers; it does
not batch decode or promise throughput. Decode/prefill fairness flags and idle
stop remain unsupported. An admin can drain a queued service through
`/v1/scheduler/drain`; `--stop-after-drain` waits for all admitted work, then
acknowledges STOP across the topology. Queue expiry or cancellation before
dispatch preserves a healthy epoch; active abandonment or failure retires it
and fails remaining waiters. `--heartbeat-interval` must be
positive and below the bounded transport timeout. PING commands between model commands share the
worker command lock and leave slot state unchanged, including while output is
backpressured; they do not take the HTTP
admission lock or cause false busy responses.

SIGTERM or SIGINT stops admission. When idle, the service sends STOP to every
worker and reports `graceful_stop: true` only after acknowledgement and cleanup.
When a request is active, shutdown interrupts transport I/O, waits for request
threads and discards the epoch; the coordinator exits 130 without a graceful
stopped event, and workers report their closed/failed transport. Cancelled SSE
has no success sentinel. Peer failure terminates the service with an error.
A newly prepared session is required after any discarded epoch.

See [glm-host-http.md](glm-host-http.md) for prepared ownership semantics. Tests
use actual local CPU subprocesses and ephemeral TLS credentials; they do not
qualify real hardware, trained tool quality or physical-network performance.
