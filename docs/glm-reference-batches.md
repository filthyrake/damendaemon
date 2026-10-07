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
