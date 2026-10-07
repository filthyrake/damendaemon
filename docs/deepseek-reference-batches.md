# DeepSeek V4 CPU reference text batches

`BatchReferenceGenerator.generate_batch` composes the prepared DeepSeek V4 model
with the same owned batch lifecycle used by Flash. It accepts 1–16 ordinary-text
conversations with equal encoded prompt lengths. Output limits can be one integer
or one integer per row, each from 1 through 4096. Thinking mode and history
reasoning options follow the existing single-request tokenizer contract.

```sh
python -m scripts.run_deepseek_reference /path/to/local/snapshot \
  --batch-messages conversations.json --context 4096 --max-new-tokens 128
```

`conversations.json` is a list of conversation arrays, bounded to 1 MiB. Ragged
encoded prompts and invalid limits/tool declarations fail before checkpoint
weight loading. `--tools tools.json` supplies one shared declaration list to the
batch; the Python interface accepts the same list through `tools=`. Existing explicit weight,
workspace and aggregate cache limits remain in force. Defaults do not admit the
full official checkpoint.

Calls are returned as data and are never executed. Each row buffers DSML calls
until its complete tool block and EOS are parsed. Undeclared names, malformed or
truncated calls fail the batch before any such call is emitted. The allowed-name
set is captured before generation, so later caller mutations cannot authorize a
new tool. Tool rows finish with `tool_calls`; other rows retain `stop`/`length`.
Tools with `strict: true` additionally enforce their parameter schema on historical
and generated calls. Other tools retain name and finite JSON-object validation.

Events include the original zero-based `row`. Every row has its own parser,
reasoning/content channels, stop condition and output limit. Finished rows are
compacted out of the shared state before the next decode, including compressed
attention and sliding-window state. Initial cache admission conservatively
reserves every row for the longest output limit.

Close abandoned iterators: cancellation applies to the whole batch. Decode,
reorder and parser errors release owned state; failed cleanup remains retryable
and blocks a new prefill until it succeeds. Final success/usage events follow
cache release and are ordered by original row. Streamed content is provisional
if another row later fails. CLI cleanup failures preserve the primary error.

Local tiny packed-checkpoint controls compare full batch logits against
independent forced decoding, including row compaction across a ratio-4 compressed
attention boundary. This does not establish official-weight semantics, ratio-128
boundary qualification, ragged/continuous HTTP batching, hardware performance or
multi-user SLOs. GitHub #58/#59 retain those broader gates.

The declared function-name set is captured after prompt validation and before
prefill in both single-request and batch generation. Mutating the caller's tools
while a stream is suspended cannot add or revoke declarations for that rendered
request. A subsequent request validates the then-current declarations afresh.
Strict schemas are also copied before prefill, so caller mutation cannot relax or
tighten an in-flight request's parameter contract.

## Strict parameter validation

Set `strict: true` inside the tool's `function` object. Its `parameters` must be an
object schema; omitted parameters mean an empty object with no extra properties.
The bounded shared subset supports object properties/required/additionalProperties,
array items, primitive types, primitive enum/const, and nested anyOf. Extra object
properties default to disallowed. Each schema is limited to 16 KiB, depth 16 and
256 nodes; unsupported keywords fail before prefill and CLI checkpoint payload
loading. Non-strict schemas retain their previous behavior.

Historical calls must satisfy the same contract before prompt encoding. Generated
calls are checked as a complete block before publication. An invalid block emits
no tool calls from that block, fails generation and releases owned state; earlier
content or another row's already returned calls cannot be retracted. No final
success event is emitted on failure. This is final-value validation, not grammar
constrained sampling: a model can still generate invalid arguments and fail the
request. Tools are never executed. CPU tests do not establish official-model tool
quality or hardware performance.


## Optional unequal-length text batches

Add `--ragged-batch` with `--batch-messages`, or pass `ragged=True` to the
ordinary Python `generate_batch` method. The default still requires equal
encoded prompt lengths. The opt-in validates every row's own prompt plus output
limit before payload loading or prefill, then groups matching prompt lengths in
first-appearance order. Original row numbers are preserved even when execution
order differs from input order.

Groups execute sequentially. Each group's cache is released before the next
allocation, so the existing cache budget covers one live group; prompts are not
padded and this is not continuous batching. All success finals wait until every
group has released state. Cancellation closes the current group without starting
the next; later failures suppress earlier success finals, and failed release
remains retryable through the existing cleanup owner.

This option currently accepts ordinary text batches only; combining it with tools
or single-conversation input fails before weight loading. Tiny loaded checkpoints
for all three families match independent per-row logits and generation/usage,
including unequal limits and shrinking rows. This adds input coverage, not GPU,
official-model, throughput or serving-concurrency qualification.
