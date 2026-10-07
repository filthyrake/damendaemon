# Flash CPU reference text batches

The explicit CPU reference batch path composes Flash's prepared mixed KDA/sparse
model state with independent ordinary-text output parsers. It supports 1–16
conversations with **equal encoded prompt lengths**. Ragged prompts are rejected;
there is no implicit padding or attention-mask approximation.

```sh
python -m scripts.run_glm_flash_reference /path/to/local/snapshot \
  --batch-messages conversations.json --context 4096 --max-new-tokens 128
```

The local JSON file is a list of ordinary text conversation arrays, for example
`[[{"role":"user","content":"Hi"}],[{"role":"user","content":"Yo"}]]`.
The file is bounded to 1 MiB. Tokenized equality and context admission happen
before checkpoint weight loading. Existing weight, aggregate cache, workspace
and output bounds apply; the defaults do not admit the full official checkpoint.

The Python `FlashBatchReferenceGenerator.generate_batch` interface also accepts
a list of output limits, one per row, each from 1 through 4096. Each event has a
zero-based `row` referring to the original input order. Reasoning/content parsing,
EOS and truncation are independent. Finished slots are compacted before the next
decode. All rows initially reserve cache capacity for the longest allowed output;
this is conservative admission, not a paged per-row allocator.

Close an abandoned iterator. Cancellation, decode errors and reorder errors
release the owned batch state. Failed releases remain retained for `cleanup()`
and block another prefill until cleanup succeeds. Final success/usage events are
emitted in original row order only after shared cache release; earlier streamed
content is provisional if a later batch error occurs. Cancellation is whole-batch,
not individual-row cancellation.

Tiny CPU fixtures cover batch logits against independent forced-decode runs,
finished-row compaction, parser isolation and lifecycle failures. This path is
not continuous batching, HTTP serving, multimodal batching, official-model
semantic qualification or a GPU performance result. Those remain open in GitHub
#57/#59. The ordinary single-request command retains its existing behavior.

The explicit [tool batch path](flash-tool-generation.md) accepts the same bounded
batch shape with shared schema declarations and independent row call buffers.


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
