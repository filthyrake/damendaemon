# T5 encoder-decoder qualification

Issue #29. Independent loader/executor; Transformers is an offline numerical
reference only. Qualified checkpoint: google/flan-t5-small revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` (Apache 2.0 model license).
Architecture/configuration reference: the pinned model's config and Hugging Face
Transformers T5 implementation (Apache 2.0). No upstream model runtime is called
by dllm. Model weights are downloaded separately, not committed.

This introduces bidirectional encoder attention, unscaled relative-bias decoder
attention, cross-attention, independent encoder/decoder lengths and gated GELU.
Six attention heads of width 64 deliberately differ from the 512-wide model.
Encoder and cross-projection work is packed; decoder slots use native CUDA
attention/projections with graph replay. Initial Apple attention uses PyTorch
SDPA and native Metal projections/norms. All weights, activations and KV are FP32.

Both A2 and Apple M4 pass 198 independently computed CPU FP32 reference checks:
encoder lengths 1/15/17/31/127/512, reordered four-slot decoding, slot replacement,
128 further decode steps, relative bucket boundaries and exact greedy output.
Maximum relative RMS: A2 0.000001725, M4 0.000000996; all selected tokens match.
The dedicated CUDA relative-attention oracle passes six dtype/head-dimension
cases with poisoned future cache entries. All 115 existing CUDA kernel checks
pass after extending the shared attention kernel.

Real HTTP proofs on both GPUs pass raw strings/token IDs, SSE/nonstream equality,
four concurrent distinct prompts, 512 encoder tokens plus output, invalid length
and chat rejection, cancellation, Qwen switching and host restoration. Unsupported
chunk/prefix options fail cleanly. Repetition penalties use decoder history,
not encoder input. The local suite passes 173 tests.

Initial serving baseline, exact shared 19-token encoder input, 8 output tokens,
512 context, four slots, five measured warm rounds after warmup:

| GPU | Concurrent requests | Output tokens/s | First text ms |
|---|---:|---:|---:|
| A2 | 1 | 241.54 | 21.79 |
| A2 | 2 | 244.43 | 42.89 |
| A2 | 4 | 416.70 | 49.11 |
| M4 | 1 | 99.18 | 15.22 |
| M4 | 2 | 179.66 | 20.34 |
| M4 | 4 | 275.72 | 31.75 |

These are dllm measurements, not competitive speedup claims. llama.cpp commit
836d57176dc699a726c55418e4f96b8ca628e1bf converts the checkpoint to FP32 GGUF,
but its four-slot server crashes in attention graph construction. Single-slot,
flash-attention-off starts but emits only two tokens, failing the matched-output
gate. The source server has no llama_encode path at this revision; its standalone
completion example does. Failed commands/logs are retained; no valid server
baseline exists yet. Do not compare those partial runs against dllm.

Initial scope: raw completions; explicit context up to configured 512; FP32;
fixed decoder and cross-attention caches. No T5 chat template, quantization,
chunked prefill, prefix snapshots or paged KV. ReLU/tied-head configurations are
implemented but require a separate real-checkpoint qualification. Other T5
variants (mT5/UMT5/LongT5), speech/vision encoder-decoders remain unsupported.

## Native Metal decoder attention

The default M4 decoder now reads its caches directly through native Metal
attention. This removes per-layer cache gathering and temporary SDPA masks.
The shared kernel supports explicit attention scaling and relative buckets;
other architectures retain their original scaled-dot-product default.

Three independent CPU attention-oracle cases (head widths 64/96/128, reordered
slots, relative boundaries and NaN-poisoned future cache) pass. All 29 existing
Metal kernel checks and 198 FLAN-T5 CPU-reference logit checks pass (maximum
relative RMS 0.000000776). Real service/residency proof also passes.

Two native serving runs, same workload as above, yield 228.96/278.06/660.90 and
225.44/401.42/575.41 tokens/s at concurrency 1/2/4. The laptop and scheduler
introduce run variation, especially at concurrency two; both runs improve over
the initial gathered-SDPA measurements. Raw runs are in metal-native/ and
metal-native-repeat/. No llama.cpp speedup claim is made.
