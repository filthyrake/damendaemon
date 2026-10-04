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

## Original T5 and A2 projection dispatch

Pinned google-t5/t5-small `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
qualifies the ReLU feed-forward, eight-head and tied-embedding/output paths.
Both GPUs pass 573 independent checks through decoder position 511, plus cache
exhaustion rejection and the real HTTP/model-switching suite. Original snapshots
contain one obsolete cross-attention relative-bias tensor; the loader validates
and discards exactly that key, matching Transformers' explicit ignore list.
Other unexpected tensors still fail loading.

Profiling the A2 FLAN-T5 four-row decoder attributes about 81% of GPU time to
FP32 projections. Geometry and tensor-core experiments are retained in the probe
scripts/evidence. cuBLAS already recovers most of the measured batched advantage,
so T5 now uses it for multiple rows while retaining the faster native vector
kernel for one row. No weight/activation precision change is made. Both T5
checkpoints pass 573 reference checks after this change, including reordered
batches and full decoder context (max relative RMS below 0.000001734).

Updated FLAN-T5 A2 HTTP throughput at concurrency 1/2/4:
258.83 / 290.84 / 527.93 tokens/s; first text 20.57 / 32.95 / 39.19 ms.
These short-output results include encoder work; the profile is not a serving
benchmark. The custom tensor-core prototype remains outside runtime dispatch.

## Valid standalone llama.cpp comparison

The public llama.cpp API supports encoder-decoder execution even though the
server path above fails. `scripts/bench_t5_llama.cpp` is an independent adapter
outside the upstream checkout, compiled against pinned llama.cpp. It calls
`llama_encode` followed by decoder-start and greedy `llama_decode` steps.
Compare `scripts/bench_t5_executor.py`; both use the exact 19 encoder IDs,
8 outputs, FP32 weights/KV, context 512, one slot, A2 only and five warm trials.
All output IDs agree. Timings exclude model load and HTTP/tokenizer overhead.

| Core inference | First token ms | Total for 8 outputs ms | Subsequent token ms |
|---|---:|---:|---:|
| dllm | 16.426 | 25.800 | 1.339 |
| llama.cpp | 6.155 | 26.806 | 2.950 |

Dllm has substantially slower encoder/prefill latency in this workload, while
its decoder is faster. Total times are close; this is not a universal win or an
HTTP comparison. Raw core trials and llama.cpp initialization log are retained.

## Opt-in CUDA prompt graphs

`DLLM_T5_PREFILL_GRAPHS=1..8` enables a bounded LRU of captured prompt shapes
and slot assignments. Default 0 retains eager prompt execution. Input token IDs
are replaced on replay; this caches GPU execution, not prompt outputs. Decoder
and cross-attention state are recomputed for every request. `--no-graph` disables
both prompt and decode capture. Apple continues eager prompt execution.

Full 573-check CUDA reference and real service proof pass with four entries.
A separate CPU-reference oracle changes token IDs at the same shape, interleaves
decode, changes slots, evicts entries and recaptures. Its first version exposed
an extra retained 8-MiB cuBLAS workspace for each newly created warmup stream;
reusing one stream fixes it. The repeated-eviction trace now stays below 9 MiB
above decode allocations for two entries rather than growing past 40 MiB.

FLAN-T5 A2 HTTP throughput becomes 428.46 / 607.85 / 1020.21 tokens/s at
concurrency 1/2/4, first text 7.25 / 10.21 / 12.76 ms. Core first token averages
3.014 ms and total eight-output time 13.020 ms with identical IDs, compared with
16.426 / 25.800 ms eager and llama.cpp 6.155 / 26.806 ms from the earlier core
run. These are warm, repeated-shape measurements. Initial capture costs roughly
300-360 ms per new shape in these runs; shape churn can therefore be slower.
The setting remains opt-in. Production should choose its cache limit based on
actual prompt-shape reuse and cold latency, not these warm numbers alone.
