# GLM CPU host topology completion

`python -m scripts.run_glm_host_topology` is the explicit GLM counterpart to the
Qwen CPU topology command. It accepts only `worker` and `complete` roles, an
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
The `ready` event identifies `native_chat`. Arbitrary templates and the `serve`
role remain unsupported. GLM projections retain the supported packed FP8 source values with
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
