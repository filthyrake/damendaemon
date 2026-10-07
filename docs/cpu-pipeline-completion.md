# Local CPU raw completion

The explicit CPU reference command composes bounded Qwen3.5 checkpoint loading,
local tokenization and owned CPU pipeline construction:

```sh
python -m scripts.run_cpu_pipeline_reference /path/to/text-snapshot \
  --prompt 'The next item is' --quantization q4_0 \
  --context 512 --max-new-tokens 32 --prefill-chunk 64 --stages 2
```

Use `--tokens prompt.json` for a local JSON list of token IDs. This is raw
completion, without a chat template. It does not admit chat, tools or images,
select a GPU, register a server model or download anything. The source must meet
[CPU checkpoint admission](cpu-hybrid-checkpoint.md). Tiny fixture results do not
qualify official-model quality, practical contexts or performance.

The tokenizer must have a contiguous mapped vocabulary no larger than the model
head. Its JSON is capped at 32 MiB. Truncation and padding are disabled; raw text
uses the tokenizer's configured special-token postprocessor. Prompt text/token
JSON and generation metadata are each bounded at 1 MiB. EOS IDs come from local
`generation_config.json` when declared, otherwise from model config; omitted EOS
means length-limited output. Invalid IDs and known/configured media tokens reject.
Prompt plus reserved output must fit context before any weight payload is read.

Generation is greedy, with output/chunk limits of 4096 tokens
and up to 16 stages. Padded vocabulary rows and known media IDs cannot be sampled.
Every emitted `token_id`, including EOS, counts toward completion usage. Text is
decoded with special tokens omitted only at the final event, after successful
pipeline cleanup. Earlier token events are provisional if later execution fails.
The opening event explicitly identifies CPU reference scope and unqualified status.

Weight, loader workspace, per-stage total, cache, execution workspace, projection
workspace and output limits are explicit CLI options, each capped at 2048 MiB.
Source storage remains owned by the prepared model; stage budgets are additional
capacity allowances rather than a measurement of free system memory. Metadata,
Python/allocator overhead and bounded tokenizer/output-event storage are separate.
No kernel timing or performance verdict is emitted.

`CpuPipelineReferenceGenerator` is also usable with a prepared model. It owns a
fresh pipeline per request or batch, rejects concurrent use, forwards cancellation through
chunked prefill/decode and closes on completion, error or generator close. Consume
its generator inside `contextlib.closing` when stopping early. Failed cleanup
retains `runner.pipeline`; retain the runner and retry `runner.cleanup()`. A primary
execution error is preserved if cleanup also fails. No successful final event is
returned until cleanup succeeds. This synchronous command does not provide HTTP
serving, batch scheduling, distributed provisioning or chat-template qualification.

## Bounded batches

`--batch-inputs requests.json` accepts a JSON list of 1..4 objects. Each has exactly
one `prompt` string or `tokens` list and an optional `max_new_tokens` integer;
the CLI output limit is the default when omitted. For example:

```json
[
  {"prompt": "The next item is", "max_new_tokens": 16},
  {"tokens": [1, 2, 3], "max_new_tokens": 8}
]
```

The entire batch is tokenized and context-checked before weight payloads. Prefill
runs in bounded chunks independently per slot, so prompt lengths may differ;
no padding is inserted. Decode batches only unfinished slots and retains their
original `row` IDs as other requests finish. Each row has its own EOS/length
reason and usage. All successful final events wait until the shared pipeline
closes. A failure or cancellation cancels the whole batch; token events already
emitted do not establish success. This is a bounded local batch, without
continuous arrival scheduling or independent per-row cancellation.

The Python equivalent is `runner.generate_batch(token_rows,
max_new_tokens=[16, 8], chunk_size=64)`. Single-request generation shares the same
ownership path and retains its previous event format. Tests compare full batched
decode logits against independent pipelines with ragged contexts and shrinking
slot lists in FP32, Q4 and Q8, alongside EOS and failure/cleanup controls.
