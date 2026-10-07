# GLM-5.3 sparse-attention CPU text batches

`dllm.glm_batch_generation.BatchReferenceGenerator` adds bounded conversation
batches to the packed GLM sparse-attention CPU reference. It reuses the shared
owned batch lifecycle already used by Flash and DeepSeek, with GLM's tokenizer,
reasoning/content parser and mapped vocabulary policy.

```sh
python -m scripts.run_glm_reference /path/to/local/snapshot \
  --batch-messages conversations.json --context 4096 --max-new-tokens 128
```

The JSON file is at most 1 MiB and contains 1–16 conversation arrays. Encoded
prompt lengths must match; ragged inputs are rejected before loading checkpoint
payloads. `--reasoning-effort` and `--clear-history-reasoning` retain the single
request behavior. Existing explicit weight/workspace/cache limits still apply.
The Python `generate_batch` interface also accepts one output limit per row,
each 1–4096. CLI `--max-new-tokens` applies to every row.

Every row independently tracks reasoning/content, EOS and output limits. Events
include the original zero-based `row`, even after finished rows are compacted out
of the sparse attention caches. Unmapped padded vocabulary rows cannot be sampled.
Cache admission reserves the whole batch through the longest output limit.

Close abandoned iterators. Cancellation applies to the entire batch; decode,
reorder or parser failures release the owned state. Failed cleanup stays reachable
and blocks another prefill until retry succeeds. Final success and usage events
are emitted only after cache release, ordered by original row. Previously streamed
text remains provisional if a later row fails.

A packed checkpoint fixture's full logits match independent floating single-row
controls through prefill and survivor compaction, using independent FP8 decoding.
CLI output with a controlled synthetic head matches single-request streams. These
are CPU correctness/lifecycle results, not trained-model quality, hardware
performance, practical long-context qualification or concurrent HTTP serving.
Tool calls and multimodal inputs remain outside this text-only adapter. Parent #56.

## Bounded text tools

`--tools tools.json` enables the explicit schema-bound tool path for either
single conversations or equal-length batches. Python callers use
`GlmToolReferenceGenerator.generate_tools` or `generate_batch_tools` with a
`GlmToolText` tokenizer. Ordinary `chat_tokens` continues to reject tool history.

The supported prompt and wire subset was independently checked against
`zai-org/GLM-5.3` revision `aca966e4e02791568aa6a4ced368624b3d897f42`:
all six full tool-history prompts and official tokenizer-ID sequences match the
separately pinned `glm53_tool_prompt_contract.json` fixture. Its template SHA256 is
`3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c`.
It matches the existing Flash text/tool subset, allowing shared rendering,
codec/parser and owned generation while keeping explicit model/tokenizer bindings.
Source licenses remain under `docs/licenses/`.

Declarations contain 1–64 named functions with a supported closed-object parameter
schema. Top-level argument types must be explicit: the wire writes string values
raw and other values as JSON, so ambiguous top-level unions are rejected. The
bounded shared schema rules apply even when `strict` is omitted/false; there is
no permissive untyped wire fallback. Nested schemas, enum/const and required
properties are checked. Unsupported fields, including deferred-loading tool
metadata, fail explicitly. No tools are executed.

Historical calls require distinct IDs and complete matching result messages.
Results render in call order; reasoning retention follows the pinned template.
Generated calls stay buffered until a complete schema-valid call block and stop
marker are parsed. Calls are returned atomically with `finish_reason: tool_calls`.
Malformed, undeclared or truncated calls fail with no calls from that block and no
final success. Other batch rows retain their original identity and stop reasons;
earlier streamed content remains provisional. Declaration snapshots cannot be
changed by caller mutation while streaming.

Bad declarations/history are rejected before CLI weight payload loading. Existing
context, input bytes, cache and workspace limits remain unchanged. Deferred tools,
media, HTTP serving and official-model tool quality remain separate qualifications.


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

The option also supports native tool batches through `--tools`; single-conversation
input still rejects the flag before weight loading. GLM/Flash Python callers pass
`ragged=True` to `generate_batch_tools`; DeepSeek uses `generate_batch(..., tools=..., ragged=True)`.
Each row retains its own parser, call arguments, stop reason and usage. Call IDs
remain scoped to their conversation row. Calls are returned, never executed;
earlier streamed content/calls remain provisional until batch success finals.
All histories and declarations validate before prefill, and parser buffers close
on abandonment or failure. Pinned prompt/wire grammars are unchanged.

Tiny loaded checkpoints
for all three families match independent per-row logits and generation/usage,
including unequal limits and shrinking rows. This adds input coverage, not GPU,
official-model, throughput or serving-concurrency qualification.
