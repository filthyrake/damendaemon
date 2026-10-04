# Expanded model and workload validation

This report records the dense-CUDA milestone. Subsequent Qwen3.5 and Apple
Silicon implementation and evidence are in [hybrid and MPS validation](hybrid-and-mps.md).
The support boundaries below describe this earlier measurement set.

A2 measurements from 2026-10-03. All GPU work selected the A2 UUID; the A30
remained assigned to its existing workload. The runtime still has no vLLM or
Transformers inference dependency. Transformers is used only for independent
validation, and vLLM runs in a separate benchmark environment.

## Architecture coverage

The shared decoder now distinguishes dense Qwen2, Qwen3, and standard-RoPE Llama
configurations. It validates head geometry and rejects unsupported variants
before loading weight tensors. Qwen3 adds bias-free projections, independent
head dimensions, and per-head Q/K normalization fused with rotary embedding and
cache writes. The 0.6B model's head dimension is 128 despite hidden-size/head-count
being 64; assuming these are equal produces an invalid implementation.

Qwen3 uses FP32 residual accumulation and FP32 normalization, rotary, and SwiGLU
intermediates. Weights, matrix operations, query outputs, and KV storage remain
FP16. This is a mixed-precision path, not all-FP32 inference. Qwen2 and Llama
retain their prior FP16 rounding behavior. The fallback tensor implementation
follows the same model-specific precision policy.

The CLI supports `--thinking auto|on|off`. The API accepts the narrowly validated
`chat_template_kwargs.enable_thinking` boolean. Only Qwen3 accepts explicit
thinking controls; other models reject them. Model-generated thinking remains
ordinary output content. Template-token equality is checked against the
independent tokenizer with Unicode, multiple chat turns, and all three modes.

Qwen3.5 is deliberately rejected: its [official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
contains recurrent linear-attention layers, gated full attention, and partial
rotary embeddings. Those need their own state and execution path. Supporting
the Qwen3 name does not imply support for Qwen3.5, MoE, vision, or quantization.

## Immutable snapshots and qualification

| Model | Revision | Expanded validation |
| --- | --- | --- |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Single, batched, and 2,048-context gates passed against FP32 reference |
| HuggingFaceTB/SmolLM2-360M-Instruct | `a10cc1512eabd3dde888204e902eca88bddb4951` | Single, batched, and 2,048-context gates passed against FP16 reference |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | Single and batched gates passed against FP32; long-context qualification remains provisional |

The original Qwen2.5 models remain in the regression suite. Configuration support
is broader than this small verified set; untested models are not automatically
qualified. These are numerical implementation checks, not broad reasoning,
knowledge, instruction-following, or safety evaluations.

## Why the reference precision matters

The original gate required exact greedy-token agreement, relative RMS logit error
below 1%, and maximum absolute error below 0.5 against FP16 Transformers SDPA.
The first Qwen3-0.6B implementation matched every greedy token but failed the
relative-error gate at one short-prompt position (1.783%). That failure remains
in [the original result](evidence/models/qwen3-small-verify_qwen2.json).

A [separate precision diagnostic](evidence/models/qwen3-small-diagnose_precision.json)
compared both implementations with FP32 Transformers on the same tokens. At the
problematic position, the original dllm output was 0.530% from FP32, while the
FP16 reference itself was 2.019% from FP32. Neither FP16 kernel ordering nor FP16
SDPA can be treated as exact arithmetic. This explains that case; it does not
excuse every numerical failure.

New Qwen3 qualification therefore explicitly uses an FP32 independent reference,
with the same 1% relative / 0.5 absolute limits and exact greedy-token requirement.
This is a reference-precision change, not a claim that the old FP16 gate passed.
Both the reference dtype and raw errors are recorded. Longer batched workloads
also exposed genuine accumulation drift, motivating the more accurate Qwen3
intermediates described above.

For the final 1.7B path, all 84 long-context checks passed, with maximum relative
error 0.9421%. SmolLM2 passed all 82 checks, with maximum relative error 0.6202%.
The difference in counts comes from Qwen3's extra thinking-template checks.

**Qwen3-0.6B is not fully qualified.** All 84 stress-test greedy choices matched,
but two positions exceeded the relative-error gate: 1.1941% and 1.4159%, with
maximum absolute errors 0.1489 and 0.1731. Its report has `passed: false`; its
performance data must not be presented as evidence that this numerical issue
is resolved. No tolerance was relaxed to make those positions pass.

Final numerical evidence:

- Qwen3-1.7B: [single](evidence/models/qwen3-medium-precise-verify_qwen2.json), [batch](evidence/models/qwen3-medium-precise-verify_batch.json), [stress](evidence/models/qwen3-medium-precise-verify_stress.json).
- SmolLM2: [single](evidence/models/smollm2-verify_qwen2.json), [batch](evidence/models/smollm2-verify_batch.json), [stress](evidence/models/smollm2-verify_stress.json).
- Qwen3-0.6B: [single](evidence/models/qwen3-small-precise-verify_qwen2.json), [batch](evidence/models/qwen3-small-precise-verify_batch.json), [failed stress qualification](evidence/models/qwen3-small-precise-verify_stress.json).

## More demanding workloads

The stress test uses different prefixes of 31, 255, 257, and 1,023 tokens in
reordered slots, then reuses two slots with 2,032 and 65 tokens while retaining
the other caches. It shrinks/reorders batches repeatedly, crosses attention-split
boundaries, and verifies that an oversized prompt leaves cache lengths intact.
Each comparison covers the full vocabulary against teacher-forced independent
reference output. Full results remain available even if the final gate fails.

Kernel checks separately verify Q/K normalization and RoPE in packed and slotted
layouts, unchanged V cache writes, FP32-intermediate RMS/SwiGLU, and masked-out
future cache entries poisoned with NaNs. Mock tests cannot replace these GPU
checks.

The real HTTP matrix runs Qwen2 → Qwen3 → Llama twice, with one and three resident
GPU models. It checks mixed-model concurrent requests, four same-model requests,
Qwen3 thinking controls, cancellation, unsupported-option rejection, recovery,
and a longer multilingual chat after failures. It uses the actual serving CLI
and stops its own server processes afterward.

[Final HTTP matrix](evidence/models/model-service-matrix-final.json),
[74 GPU kernel checks](evidence/models/kernels-multimodel.json),
and original Qwen2.5 regressions ([0.5B](evidence/models/regression-a.json),
[1.5B](evidence/models/regression-b.json)) passed. All 37 local tests passed;
those cover configuration rejection, API controls, and lifecycle behavior,
and are separate from GPU numerical proof. The [Qwen3 tensor-backend batch
check](evidence/models/qwen3-medium-fallback-final.json) also passed against FP32.

## Matched HTTP performance

Each engine uses the same pinned model, FP16 weights/KV, greedy decoding,
2,048-token context capacity, four active slots, no prefix cache, and four CPU
threads. Qwen3 thinking is disabled on both sides. dllm's FP32 residual/elementwise
policy differs from vLLM's internal precision choices and is part of the tested
implementation, not an all-operations precision match.

The common client requests 64 output tokens; usage must confirm that count and
the stream must complete. Six rounds are run at each concurrency, excluding the
first warmup. Aggregate output rate includes prefill and HTTP delivery; the
post-first-output rate is per request. These short shared-VM samples do not
establish tail latency or a general engine ranking.

| Model | Engine | Concurrent requests | Aggregate tokens/s | Per-request post-first tokens/s | First output (ms) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | dllm | 1 | 45.0 | 45.3 | 30.2 |
| Qwen3-1.7B | vllm | 1 | 45.2 | 46.1 | 48.0 |
| Qwen3-1.7B | dllm | 4 | 151.2 | 39.3 | 91.2 |
| Qwen3-1.7B | vllm | 4 | 161.6 | 42.3 | 91.7 |
| SmolLM2-360M | dllm | 1 | 137.6 | 151.6 | 49.9 |
| SmolLM2-360M | vllm | 1 | 159.7 | 176.5 | 43.6 |
| SmolLM2-360M | dllm | 4 | 423.7 | 129.2 | 89.4 |
| SmolLM2-360M | vllm | 4 | 557.5 | 166.3 | 75.1 |
| Qwen3-0.6B (provisional) | dllm | 1 | 108.2 | 112.9 | 31.1 |
| Qwen3-0.6B (provisional) | vllm | 1 | 113.6 | 121.1 | 42.7 |
| Qwen3-0.6B (provisional) | dllm | 4 | 325.7 | 88.5 | 72.1 |
| Qwen3-0.6B (provisional) | vllm | 4 | 384.4 | 107.1 | 73.4 |

Raw measurements (including concurrency 2):

- Qwen3-1.7B: [dllm C1](evidence/models/compare-dllm-qwen3-medium-c1.json), [dllm C2](evidence/models/compare-dllm-qwen3-medium-c2.json), [dllm C4](evidence/models/compare-dllm-qwen3-medium-c4.json), [vllm C1](evidence/models/compare-vllm-qwen3-medium-c1.json), [vllm C2](evidence/models/compare-vllm-qwen3-medium-c2.json), [vllm C4](evidence/models/compare-vllm-qwen3-medium-c4.json).
- SmolLM2-360M: [dllm C1](evidence/models/compare-dllm-smollm2-c1.json), [dllm C2](evidence/models/compare-dllm-smollm2-c2.json), [dllm C4](evidence/models/compare-dllm-smollm2-c4.json), [vllm C1](evidence/models/compare-vllm-smollm2-c1.json), [vllm C2](evidence/models/compare-vllm-smollm2-c2.json), [vllm C4](evidence/models/compare-vllm-smollm2-c4.json).
- Qwen3-0.6B (provisional): [dllm C1](evidence/models/compare-dllm-qwen3-small-c1.json), [dllm C2](evidence/models/compare-dllm-qwen3-small-c2.json), [dllm C4](evidence/models/compare-dllm-qwen3-small-c4.json), [vllm C1](evidence/models/compare-vllm-qwen3-small-c1.json), [vllm C2](evidence/models/compare-vllm-qwen3-small-c2.json), [vllm C4](evidence/models/compare-vllm-qwen3-small-c4.json).

Prompt-token counts matched for each model/engine pair (26 for Qwen3, 43 for
SmolLM2); all measured completions contained 64 tokens. Text is recorded in
every trial. Qwen3 continuations can vary with batch shape and engine arithmetic;
these are equal-length workloads, not a guarantee of identical continuation
tokens or batch-invariant output for all prompts.

On this workload, Qwen3-1.7B aggregate throughput is close at one request and
about 6% below vLLM at four requests. SmolLM2 is about 24% below at four requests.
The stronger Qwen3 precision policy preserved useful throughput,
but smaller models still expose substantial overhead. The
[SmolLM2 CUDA profile](evidence/models/smollm2-profile.json) records the remaining
kernel costs for targeted follow-up: 425 kernels/token and 5.49 ms of summed
GPU kernel time/token. Its three largest matrix-operation kernel groups account
for about 89% of that GPU time; launch/HTTP overhead is additional. Profiled
timings are diagnostic and are not substitutes for the unprofiled rates above.
These are observations
from this test, not claims about all workloads or model quality.

## Reproduce

Use the pinned model metadata in `evidence/models/` and the README's isolated
CUDA environment. Select one GPU UUID and set `DLLM_KERNELS=triton`:

```sh
PYTHONPATH=. python scripts/verify_qwen2.py /path/to/qwen3 --reference-dtype float32
PYTHONPATH=. python scripts/verify_batch.py /path/to/qwen3 --reference-dtype float32
PYTHONPATH=. python scripts/verify_stress.py /path/to/qwen3 --reference-dtype float32
PYTHONPATH=. python scripts/verify_stress.py /path/to/smollm2
PYTHONPATH=. python scripts/verify_kernels.py
PYTHONPATH=. python scripts/verify_model_service.py /path/to/qwen2 /path/to/qwen3 /path/to/smollm2
```

The A2-specific benchmark launcher accepts `--model-metadata`, `--label`,
`--context`, and `--thinking`. It launches engines sequentially and terminates
only its own process groups. It assumes the existing separate benchmark venv.
