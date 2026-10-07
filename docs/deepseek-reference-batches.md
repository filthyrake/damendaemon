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
This preserves the existing codec scope: JSON object arguments and declared
names are validated; arbitrary supplied parameter JSON Schemas are not enforced.

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
This is name/lifetime enforcement, not parameter JSON-schema validation.
