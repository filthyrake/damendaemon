# Mamba-1 state-space inference

Issue #30. Qualified checkpoint: state-spaces/mamba-130m-hf revision
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1`. Weights are downloaded separately.
Configuration/checkpoint source: https://huggingface.co/state-spaces/mamba-130m-hf
Algorithm references: the Mamba selective state-space formulation and Hugging
Face Transformers Mamba implementation (Apache 2.0). dllm owns its loader,
convolution, recurrence, batching and residency; no Transformers model, vLLM,
mamba-ssm or causal-conv1d runtime is used during production inference.

Native CUDA and Metal kernels implement causal depthwise convolution, SiLU,
input-dependent time-step discretization, selective recurrent state updates and
gated output. FP32 weights, residuals, convolution history and SSM state. State
storage is fixed per request slot rather than growing with context. CUDA decode
uses graph replay; Apple uses native Metal projection/norm/recurrence kernels.
Prompt sequences are initially processed separately to avoid padding advancing
recurrent state. This is an optimization opportunity, not packed-prefill support.

Both GPUs pass 12 independent CPU kernel-oracle cases spanning odd widths,
state sizes 1/16/32, convolution widths 1/4/8, short prompts, reset, continued
updates and reordered slots. Both pass 103 full-model independent CPU FP32 logit
checks at contexts 512 and 2048, plus convolution/SSM state comparisons, exact
free-running greedy IDs, replacement/reordering and exhausted-context rejection.
Maximum logit relative RMS: A2 0.00002468, M4 0.00001579. Maximum absolute logit
error at 2048: A2 0.00047684, M4 0.00065613. Short prompts expose an HF reference
pitfall: its cache_position refers to convolution ring insertion, not raw prompt
length. The validator follows generate() and advances from conv_kernel, including
for one-token prompts. The independent CPU recurrence oracle separately verifies
this behavior without using Transformers cache helpers.

Real HTTP tests on both GPUs pass raw strings/token IDs, SSE/nonstream equality,
four concurrent distinct prompts, full prompt/output context, invalid requests,
cancellation and switching to Qwen and back. Prefix/chunk options are explicitly
rejected. Local suite: 174 passing tests.

## Matched serving baseline

Pinned checkpoint, FP32 weights and state, context 512 per request, the exact
shared token IDs in mamba-prompt.json, 32 output tokens, five warm measured
rounds. llama.cpp commit 836d57176dc699a726c55418e4f96b8ca628e1bf with FP32 GGUF;
vLLM 0.30 float dtype. Engines run sequentially on each GPU. All trial texts agree
across every engine/device/concurrency. These are serving results, not kernel
microbenchmarks or quality comparisons.

| GPU / engine | c1 tokens/s | c2 tokens/s | c4 tokens/s | c1 first text ms |
|---|---:|---:|---:|---:|
| A2 dllm | 239.97 | 277.28 | 446.14 | 21.32 |
| A2 llama.cpp | 156.88 | 190.39 | 221.33 | 26.26 |
| A2 vLLM | 191.41 | 265.57 | 453.83 | 49.54 |
| M4 dllm | 53.33 | 94.25 | 182.89 | 28.36 |
| M4 llama.cpp | 94.14 | 173.02 | 239.66 | 28.45 |

Apple remains materially behind llama.cpp; A2 is slightly below vLLM at
concurrency four in this run. More measurements/optimization are required.
The host profile separates dispatch/readback and explicitly does not attribute
GPU waits to Python execution.

Initial scope is Mamba-1 raw text generation with fixed state and context up to
2048. Quantization, prefix snapshots, chunked prefill, speculative decoding,
Mamba2, Falcon-Mamba/H1, multimodal input and arbitrary longer contexts are not
qualified by this checkpoint. Chat requests are rejected without a template.

## Packed projection reads and experimental Apple fusion

Native convolution and recurrence now read signal/gate data directly from the
packed input projection, removing the two copies needed for batched rows. The
expanded kernel oracle verifies both packed and separate layouts against CPU,
including state contents. Both devices retain their full 2048-context reference
qualification. A2 packed serving measures 236.04 / 289.19 / 483.76 tokens/s at
concurrency 1/2/4, versus initial 239.97 / 277.28 / 446.14 in the earlier run.

Metal also implements a fused time-step projection plus recurrence. It is
**experimental and off by default**: `DLLM_MAMBA_FUSED_DT=1` enables it. The fused
CPU kernel/model/state oracles pass, but isolated operation gains do not establish
a full serving improvement. Counterbalanced isolated medians in microseconds:

| Rows / inner width | Separate projection/update | Fused |
|---|---:|---:|
| 1 / 1536 | 117.85 | 57.63 |
| 4 / 1536 | 93.40 | 68.40 |
| 1 / 2048 | 146.36 | 65.42 |
| 4 / 2048 | 121.86 | 87.52 |

Full laptop serving varies strongly: fused runs give 119.72/214.66/266.76 and
64.42/91.06/332.67 tokens/s, while the repeated llama.cpp run gives
141.35/240.12/340.94. Same-executor alternating core decode also fails to show a
stable single-row gain (unfused/fused median 113.55/110.05, then 81.97/73.32 after
explicit warmup). Four-row medians are 406.98/413.04 and 314.91/346.84. Consequently
fusion is not selected automatically. Raw rounds are retained rather than
presenting the best run as the default performance. Apple remains a performance
work item; the A2 result cannot stand in for it.

## Larger Mamba checkpoint

Pinned state-spaces/mamba-370m-hf
`b519127f5bfaaa1c27dd938dad051ec360972b23` expands coverage to 48 layers,
1024 hidden width, 2048 intermediate width and rank-64 time steps. A2 and M4
pass the full 2048-context logits/state trace and real service/model switching
(the Apple long trace exercised experimental fusion; the service proof uses the
default separate projection). Max relative RMS is 0.00001412 on A2 and
0.00005418 on M4. Raw results are in 370m/.

Same A2 FP32 serving method/input/32 outputs as above:

| Engine | c1 tokens/s | c2 tokens/s | c4 tokens/s |
|---|---:|---:|---:|
| dllm | 92.45 | 106.30 | 183.57 |
| llama.cpp | 70.79 | 77.25 | 83.79 |
| vLLM | 87.31 | 97.28 | 186.83 |

These compare installed configurations. vLLM reports no A2-specific selective
state-update tuning file and uses its default configuration; the results do not
claim to beat every manually tuned vLLM setup. More/larger Mamba checkpoints
remain separate qualification work.
