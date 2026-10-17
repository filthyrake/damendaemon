# Flash CPU reference tool generation

The explicit `FlashToolReferenceGenerator.generate_tools(messages, tools, ...)`
path connects the pinned tool-history adapter to bounded, schema-aware incremental
output. Calls are returned as data and are never executed.

```sh
python -m scripts.run_glm_flash_reference /path/to/local/snapshot \
  --messages conversation.json --tools functions.json --max-new-tokens 512
```

Both JSON files are local and bounded to 1 MiB. Declarations/history/context are
validated before checkpoint payload loading. Existing weight, cache, workspace
and output budgets remain explicit; defaults do not admit the full official
checkpoint. `--batch-messages conversations.json --tools functions.json` also
accepts 1–16 equal-token-length tool histories with a shared declaration snapshot.
The separate [still-image/tool path](flash-image-tools.md) composes image inputs
with single-request tool output. Ordinary text, ordinary batch and still-image
commands retain their admission.

Reasoning and ordinary content stream incrementally. Tool blocks are buffered
until complete, terminated by a supported model stop marker, and validated as a
whole against the declaration snapshot. Only then is a `tool_calls` event returned.
The final finish reason is `tool_calls`; ordinary replies retain `stop`/`length`.
JSON-looking strings retain their declared string type. Truncated/undeclared or
malformed calls fail, including a partial tool opener at truncation. A call cannot
begin before reasoning closes. Retained completion text is bounded to 1 MiB.

Use returned call IDs in subsequent `tool_call_id` results. The next history must
include every result exactly once. The caller supplies those results; damendaemon does
not invoke functions. Close abandoned iterators. Parser/decode/cancellation paths
release cache ownership and close the output buffer; failed cache release stays
owned for retry. Final success follows release. Earlier deltas/call data remain
provisional if a later cleanup error prevents the final event.

CPU tests cover every character split, UTF-8 token boundaries, all supported stop
markers, declared argument types, result-history continuation, malformed/truncated
output, context admission and cleanup failure. A tiny loaded snapshot exercises
the CPU command, and scripted logits control the tool replies. These tests do not
establish official-model tool quality, hardware serving or performance. GitHub
#57 retains those broader qualification gates.


`generate_batch_tools` accepts one output limit or one per row. Each event keeps
its original `row`; parsers, argument types and call buffers remain independent
as finished rows are compacted. Mixed tool/text rows use `tool_calls`, `stop` or
`length` separately. Call IDs are local to each row and must be paired with that
row's history when supplying results. Cancellation applies to the whole batch;
any parser/decode/reorder failure closes all output buffers and the shared cache.
Final metadata is emitted only after cleanup. The CLI rejects ragged/context or
history failures before checkpoint payload loading.
