# Qwen3.5 and Apple GPU qualification

This extends the earlier [dense CUDA matrix](model-matrix.md). Results describe
specific checkpoints, hardware, precision, and workloads; they do not establish
full vLLM or llama.cpp parity. GitHub issues #1–#4 track the remaining work.

## Qwen3.5 text on NVIDIA A2

The independent hybrid executor supports the dense text backbones of:

| Checkpoint | Immutable revision |
| --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` |

Execution uses FP16 projection weights and attention KV, FP32 residuals and
recurrent state, causal convolution, gated full attention, and partial RoPE.
The vocabulary projection accumulates and returns FP32 logits. Each request
slot owns its convolution, recurrent, and attention state. CUDA graphs capture
decode batches; constructor warmup state is cleared before serving requests.
Vision and multi-token prediction weights are excluded from text loading.
Multimodal input, MoE, and quantized variants remain unsupported.

The production environment is PyTorch 2.6.0+cu124. A separate environment with
Transformers 5.18.0 and PyTorch 2.13.0+cu130 generates independent FP32 reference
logits. `verify_stress.py --golden-output` records teacher-forced inputs and
reference logits; `replay_reference.py` validates the production environment
without importing Transformers. Golden tensors are local generated artifacts,
not downloaded executable model code.

The gate requires exact greedy-token agreement, relative RMS logit error below
1%, and maximum absolute error below 0.5. Both checkpoints pass 84 comparisons
at context 2048: mixed prompt lengths, 255/257 boundaries, a 2032-token prompt,
reordered and shrinking decode batches, slot reuse, and three template modes.
Worst production replay relative RMS error is 0.2155% for 0.8B and 0.3094% for
2B. The live service also passes mixed-model requests, capacity-one eviction,
resident switching, cancellation, rejected requests, and subsequent recovery.

Earlier failed experiments are retained in `evidence/hybrid`: the initial
0.8B precision path exceeded the error gate, and a later path lost a close
greedy-token ordering through FP16 output rounding. The final `*-native-replay`
and `*-stress-final` files are the qualification evidence. These failures were
fixed through arithmetic changes, without relaxing tolerances.

### Optional parallel prefill

`DLLM_HYBRID_PREFILL=fla` uses the optional `fla-core==0.4.2` kernel package
for prompts of at least 64 tokens. The default independent scan needs no FLA.
Both FLA replays pass the same 84 comparisons, with worst relative RMS errors
of 0.2155% and 0.4283%. FLA owns only the selected prefill kernel; dllm retains
weights, state, decoding, scheduling, and model switching.

Warm median prefill times on the A2, four measured repetitions after one
excluded warmup:

| Model | Prompt lengths | Independent scan | FLA | Reduction |
| --- | --- | ---: | ---: | ---: |
| 0.8B | 31, 255, 257, 1023 | 468 ms | 411 ms | 12% |
| 0.8B | 2032, 65 | 571 ms | 517 ms | 9% |
| 2B | 31, 255, 257, 1023 | 708 ms | 655 ms | 7% |
| 2B | 2032, 65 | 861 ms | 812 ms | 6% |

Compilation can be much slower than warm inference; first use is excluded
from these numbers and remains in the evidence. Hybrid prompts are currently
prefilled independently, so this does not measure packed or chunked prefill.
The optional kernel library is [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention),
under its MIT license. It is installed as a dependency, not vendored.

## Apple M4, 32 GiB unified memory

The dense backend is explicitly selected with `--device mps`. It uses FP32
device weights, activations, and KV. FP16 MPS experiments failed the independent
CPU FP32 numerical gate, so those settings are not qualified. There is no
automatic CPU execution fallback. CUDA graphs are never enabled on MPS;
`graph_requested` in newer validation output distinguishes a request from actual
capture. Early experiment files predate that distinction.

PyTorch 2.13.0 compiles dllm's Metal source for RMSNorm, SwiGLU, matrix-vector
multiplication, rotary/cache writes, and decode attention. Larger prefill matrix
operations use native PyTorch MPS operations. `DLLM_KERNELS=torch` retains a
tensor baseline and `DLLM_KERNELS=metal` requires the custom kernels.

Qwen2.5-0.5B and Qwen3-0.6B pass 48 single-sequence greedy comparisons each.
SmolLM2-360M passes batched decode, reordered slots, and cache reuse, plus 82
long-context comparisons at context 2048 against CPU FP32. Qwen3-0.6B also
passes 84 expanded comparisons at that context. Native kernel tests
cover 25 groups, including attention at context 4096 and the tensor fallback
at 4097. The live three-model service passes switching, concurrent requests,
cancellation, and recovery. This Apple result does not close the separate
Qwen3-0.6B CUDA precision issue.

### Matched HTTP comparison

SmolLM2-360M-Instruct revision `a10cc1512eabd3dde888204e902eca88bddb4951`,
FP32 device weights and KV, context 512 per slot, four CPU threads, 43 prompt
tokens, and exactly 64 generated tokens. The llama.cpp GGUF was converted from
the same checkpoint. Engines run sequentially, prefix caching is disabled, and
one warmup round is excluded before five measured rounds. Values below are
medians; first-output latency is observed by the streaming HTTP client, not a
GPU-only timer. Reference llama.cpp commit:
`836d57176dc699a726c55418e4f96b8ca628e1bf`.

| Concurrent requests | dllm Metal aggregate tokens/s | llama.cpp aggregate tokens/s | dllm first output | llama.cpp first output |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 46.3 | 55.2 | 32.3 ms | 28.0 ms |
| 2 | 83.4 | 95.3 | 55.2 ms | 37.8 ms |
| 4 | 138.7 | 145.9 | 110.3 ms | 65.4 ms |

The same PyTorch 2.13 tensor baseline reaches 29.9 tokens/s at concurrency one
and 68.5 at four. Thus the native kernels improve throughput by 1.55x and 2.03x
in this workload. The older PyTorch 2.6 baseline reached 17.9 and 49.0; comparing
only against that baseline would incorrectly attribute the dependency upgrade
to custom kernels. llama.cpp still leads throughput and first-output latency.
These FP32 results do not predict quantized-model performance or larger models.

Commands, complete trial records, model output, and correctness measurements
are stored under [evidence/mps](evidence/mps) and [evidence/hybrid](evidence/hybrid).
Reproduce with `scripts/compare_mps.py`, `scripts/verify_metal_kernels.py`,
`scripts/verify_stress.py`, and `scripts/verify_model_service.py`. Dense reference
checks use the separate `validation` extra (Transformers 4.51.3); hybrid reference
checks require the newer isolated environment described above.

### Host dispatch investigation

`scripts/profile_host.py` separates unprofiled decode dispatch from blocking
token readback and records a separate cProfile call breakdown. Host API time
includes waits for GPU work; this is not a measurement of pure Python overhead.
The call profile includes prefill; the unprofiled timings exclude it.

The initial profile exposed three small metadata tensor constructions per eager
decode step. The eager path now packs token IDs, positions, and slot IDs into
one transfer. This reduces decode tensor constructions from 96 to 32 for a
32-step run, while CUDA graph metadata staging is unchanged. The updated path
passes all 82 SmolLM2 long-context checks. Before/after profiles are retained as
`metal-host-*.json`; short-run timing variability means those profiles alone
are not an end-to-end speedup claim.

## Remaining scope

Full parity is unfinished: quantization, paged KV, chunked prefill, prefix
caching, non-greedy sampling, tools, structured output, vision/audio, MoE,
additional rotary schemes, and distributed execution are absent. Hybrid Apple
execution is unqualified. Python still handles scheduling and per-token host
work; native kernels do not establish that host overhead is negligible. Kernel
profiles and end-to-end latency measurements should guide the next native
execution-loop changes.
