# AMD ROCm qualification

[Documentation](README.md) · [AMD work and acceptance gates](https://github.com/filthyrake/dllm/issues/52)

AMD is a planned execution target, **not a qualified dllm backend yet**. A ROCm
installation or successful GPU discovery does not establish model correctness,
packed-kernel support, memory safety or competitive inference performance.

## First target: HOMER

Read-only inspection on 2026-10-06 found:

| Property | Observed value |
|---|---|
| Processor / integrated GPU | Ryzen AI Max+ 395 / Radeon 8060S |
| ROCm architecture | `gfx1151` |
| Operating system | Rocky Linux from CIQ Pro AI 9.7 |
| Kernel | `6.19.8-1.el9.elrepo.x86_64` |
| System memory | Approximately 122 GiB total, 111 GiB available at inspection |
| Selected host ROCm path | `/opt/rocm` resolves to `/opt/rocm-7.2.0` |
| Other installed ROCm directories | 7.2.1 and 7.2.4; directory presence is not qualification |
| GPU access | User has access to KFD and render device |

Existing Sheena and Fuzzball services remain active. No service, driver, kernel
or global ROCm setting was changed. The inspection showed 0% GPU activity at
that instant; this is not an exclusive reservation or a benchmark baseline.

`rocm-smi` reported a 2 GiB VRAM aperture while `rocminfo` exposed a much larger
memory pool. Neither reading alone is a safe admission budget for the shared
memory GPU. Allocator behavior, host pressure and actual release must be
qualified before large-model claims. The first work is bounded isolated probes.

## Runtime boundary

PyTorch's HIP build [intentionally uses `torch.cuda` APIs](https://docs.pytorch.org/docs/stable/notes/hip.html).
Those names do not alone make an execution path NVIDIA-only. dllm nevertheless
currently requires NVIDIA-style UUID selection, has NVIDIA-specific native
libraries, and has not qualified its Triton policies on this device. AMD needs
explicit dispatch and capability checks; it must not load CUDA/Marlin binaries.

The [AMD Ryzen Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
identifies gfx1151 support for specified stack combinations. It does not certify
this installed Rocky environment or dllm. Reproducible environment pins and
actual model/performance results will be recorded here as they are obtained.

The task list lives in [issue #52](https://github.com/filthyrake/dllm/issues/52);
the project-wide completion contract is [issue #3](https://github.com/filthyrake/dllm/issues/3).

## First bounded HIP primitive result

The existing isolated Python 3.11 environment uses PyTorch
`2.11.0+rocm7.13.0a20260425` / HIP `7.13.26162`, independently of the host
`/opt/rocm` symlink. Six real-device matrix, grouped causal attention and
normalization comparisons passed against CPU FP64 arithmetic. Peak Torch
allocation was about 97 MiB. This is not model execution or a speed benchmark.

The [initial report](evidence/amd/homer-hip-primitives-initial.json) failed its
strict release gate: 32 MiB remained after the first operation, stable on the
second cycle. An [explicit workspace-release diagnostic](evidence/amd/homer-hip-primitives-workspace-release.json)
returned allocation to zero after both cycles with unchanged numerical gates.
This identifies the retained allocation as the documented
[HIP BLAS workspace](https://docs.pytorch.org/docs/2.14/notes/hip.html#hipblas-workspaces),
not a retained probe tensor. The diagnostic does not change dllm's runtime
cleanup policy; production admission must account for library workspaces.

The runtime reported roughly 61.5 GiB total device memory but roughly 111 GiB
free through its allocator API. These APU reporting differences still need an
explicit bounded admission policy; no large allocation was attempted. The
attention call used the available default path; experimental fast attention was
not enabled and no fused-attention performance claim is made.

Active model targets are **Qwen3 or newer, with Qwen3.5 prioritized**. Qwen3
dense bring-up and Qwen3.5 hybrid-state correctness are separate gates. Older
fixtures remain historical evidence and cheap regression tests, not new GPU
qualification or tuning targets.

## Qwen3.5 text baseline on the AMD branch

The experimental `codex/amd-rocm-bringup` branch adds explicit `--device rocm`,
HIP build/device checks, rejection of NVIDIA-only libraries and shared-memory
admission. Initial scope is one gfx1151 device. The default process allocation
budget is 8 GiB; admission also limits itself to reported free/total device
memory and preserves 16 GiB of live host headroom. `DLLM_ROCM_MEMORY_GIB` is
an explicit override, not permission to exceed those other bounds. This has
463 passing local regression tests; these tests do not prove GPU speed.

The pinned Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17` passes
[84 actual AMD comparisons](evidence/amd/qwen35-hip-torch-replay512.json) against
a separately generated [CPU FP32 reference](evidence/amd/qwen35-cpu-fp32-golden512.json).
There are no greedy-token mismatches; maximum relative RMS is 0.002173 and
maximum absolute logit error is 0.025072. The gate retains RMS <0.01, absolute
<0.5, finite logits and greedy agreement. It covers 512 context, reordered
batches and slot reuse, using `torch-hybrid` without graph capture.

A reporting-only bug initially interrupted serialization because hybrid
executors do not expose the dense executor's `compute_dtype` attribute. The
replay reporter now records null for that field; numerical checks are unchanged.
This result does not qualify fused kernels, quantized text, images, the HTTP
lifecycle, long context or comparative performance. Those remain issue #52/#53
gates; no old checkpoint is being substituted for the modern target.

The same 84 checks now also pass with [fused eager execution](evidence/amd/qwen35-hip-triton-replay512.json)
and [fused graph replay](evidence/amd/qwen35-hip-triton-graph-replay512.json).
Graph replay has maximum relative RMS 0.001769
and zero greedy mismatches. This establishes those paths only for the tested
Qwen3.5-0.8B FP16-weight/FP32-state configuration at context 512. The timing
fields include warmup/first-use effects and are not a matched speed comparison.

The pinned Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` also passes all
[84 fused graph checks](evidence/amd/qwen35-2b-hip-triton-graph-replay512.json)
against its [CPU FP32 reference](evidence/amd/qwen35-2b-cpu-fp32-golden512.json),
with zero greedy mismatches, maximum relative RMS 0.005288 and maximum
absolute error 0.066978. Scope remains context 512 and unquantized text.
The [ordinary HTTP lifecycle gate](evidence/amd/qwen35-hip-service512.json)
also passes for both sizes at GPU residency capacities one and two. Each
capacity runs 20 alternating cycles (41 requests including the final return),
then streaming, cancellation recovery, failed-load recovery, mixed concurrent
requests and a four-request same-model batch. Deterministic outputs remain
unchanged. Live allocation is stable after warmup. Explicit unload releases
host caches and model residency; 96 MiB of device allocation remains, identical
after a subsequent disk reload and second unload. This is stable retention,
not a claim of zero device allocations or a matched latency benchmark.

## Qwen3.5 Q8 text

Qwen3.5-0.8B also passes all [84 Q8 graph comparisons](evidence/amd/qwen35-q8-hip-triton-graph-replay512.json)
at context 512: maximum relative RMS 0.002812, maximum absolute logit error
0.030422, zero greedy mismatches. The [CPU FP32 reference](evidence/amd/qwen35-q8-cpu-fp32-golden512.json)
uses independent `gguf==0.19.0` quantization/dequantization of projection
weights. This validates packed execution against quantized coefficients; it
does not measure quality loss against original weights or competitive speed.
The reference-only dependency lives in the isolated task directory on HOMER.
The [4K Q8 graph replay](evidence/amd/qwen35-q8-hip-triton-graph-replay4096.json)
also passes all 84 checks against its [independent reference](evidence/amd/qwen35-q8-cpu-fp32-golden4096.json),
including a 4,080-token prompt and reused slots. Worst errors remain within the
same bounds above. Q8 lifecycle and larger-model quantization remain separate
gates.

## Reproducing the experimental path

Use a separate environment containing the tested HIP PyTorch/Triton stack above.
The generic Linux `runtime` extra pins a CUDA-oriented PyTorch version and is
not yet a ROCm installation recipe. Do not install it into an existing HIP
environment. ROCm packaging is still an acceptance gap.

With dllm and its tokenizer/weight-loading dependencies available in that
environment, select the one physical AMD GPU explicitly:

```sh
unset CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export DLLM_KERNELS=triton
export DLLM_ROCM_MEMORY_GIB=8

dllm run Qwen/Qwen3.5-0.8B --device rocm --context 512 \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --thinking off --prompt 'Explain matrix multiplication briefly.'
```

For the numerically checked Q8 configuration, set `DLLM_QUANTIZATION=q8_0`.
This on-load format is separate from GGUF import. Retain explicit context and
device selection; neither gfx1151 qualification nor these memory limits imply
support for other AMD devices, ROCm versions or multiple GPUs.

Qwen3.5-2B now passes all [84 Q8 graph checks at 4K](evidence/amd/qwen35-2b-q8-hip-triton-graph-replay4096.json)
against its [CPU FP32 quantized reference](evidence/amd/qwen35-2b-q8-cpu-fp32-golden4096.json).
Maximum relative RMS is 0.004964, maximum absolute error 0.049711, and all
greedy tokens match. The [quantized HTTP lifecycle gate](evidence/amd/qwen35-q8-hip-service4096.json)
also passes with 4096 context capacity, 20 alternating cycles per residency
configuration, concurrent/streaming/cancellation checks and disk reload.
Post-unload device allocation is 32 MiB and identical after reload/unload in
both configurations; host cache reaches zero. Its short chat prompts do not
replace the separate long-prompt reference proof.

## Modern performance comparator preparation

A clean llama.cpp source archive at
`836d57176dc699a726c55418e4f96b8ca628e1bf` builds with host HIP 7.2 for
`gfx1151` in the isolated task directory. Existing HOMER binaries/checkouts
are not modified. The pinned 0.8B checkpoint converts to a Q8 GGUF with FP16
input embeddings. Because upstream conversion omits the tied output tensor,
a separate comparator artifact adds an explicit Q8 head to match dllm's
projection policy. Its [byte-preservation report](evidence/amd/qwen35-q8-comparator-head.json)
checks all 335 original tensors unchanged and the added head against independent
GGUF quantization. This proves artifact preparation, not speed parity.

The fixed `scripts/benchmark_modern_http.py` workload records exact raw input
IDs, 128 generated tokens, 4096 context, 128/512/2048 input lengths, three
warmups and ten measured rounds at concurrency one/four. It also warms groups
of two/three, retains partial failures, and requires explicit model/engine/
hardware/precision manifests. Both engines receive identical EOS suppression
bias; shorter output is a failure. ABBA process orchestration, internal decode
and memory telemetry, and cross-engine precision/numerical reconciliation
remain prerequisites for a competitive verdict. Client-visible post-first-output
rate is explicitly labeled a proxy. No AMD competitive speed result exists yet.

One remaining precision distinction is convolution history: dllm retains FP16
history while upstream llama.cpp's hybrid recurrent allocation uses FP32. Do
not call those complete arithmetic policies identical merely because their
projection coefficients and KV settings match.

## First Qwen3.5 Q8 engineering comparison

The [complete ABBA summary](evidence/amd/modern-qwen35-q8-abba02/comparison.json)
uses the pinned 0.8B checkpoint and comparator above, context capacity 4096,
128 generated tokens, three group warmups and ten measured rounds per phase.
Raw per-request results, manifests and logs are retained as compressed files in
the same directory. Engine order is dllm, llama.cpp, llama.cpp, dllm.

| Input tokens | Users | dllm aggregate tokens/s | llama.cpp aggregate tokens/s | dllm p95 first output (s) | llama.cpp p95 first output (s) | Exact text across engines |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 1 | 138.37 | 153.14 | 0.027 | 0.061 | yes |
| 128 | 4 | 193.67 | 435.80 | 0.120 | 0.215 | yes |
| 512 | 1 | 131.41 | 141.39 | 0.077 | 0.127 | yes |
| 512 | 4 | 180.43 | 327.77 | 0.320 | 0.514 | no |
| 2048 | 1 | 87.59 | 99.89 | 0.587 | 0.491 | yes |
| 2048 | 4 | 106.98 | 159.73 | 2.334 | 2.056 | yes |

Throughput is the median complete-round output rate, including prompt work.
Within dllm, text stays stable across all measured rounds. llama.cpp varies in
the 512-input/four-user cell; that failure is retained, not relabeled exact
parity. These are engineering comparisons with the arithmetic distinctions
above, not a competitive pass. Existing host services were preserved; the host
was not exclusively reserved. Internal decode/peak memory gates remain open.

The clear actionable gap is concurrent decoding: four-user scaling falls well
behind the comparator, while single-user throughput is closer. The next bounded
hypothesis under #54/#59 is the AMD small-batch Q8 projection policy, currently
using tiles selected from A2 measurements. No optimization is promoted from this
diagnosis alone.

The [product context minimum is 32K](direction.md). This 4K experiment is a
development baseline and does not qualify long-context multi-user service.

## Isolated gfx1151 small-batch result

The [within-dllm ABBA comparison](evidence/amd/modern-qwen35-q8-small-batch-abba01/comparison.json)
changes only FP16 Q8 projection dispatch for two through four rows: use the
existing multi-row matvec instead of the A2-tuned tiled matrix kernel. Ordinary,
candidate, candidate, ordinary phases use identical checkpoint, arithmetic and
storage policy, hardware, prompts, output length and context. Raw reports,
manifests and logs are retained beside the summary. This measures the isolated
experiment at revision `6c709a9`, not a change to the production default.

| Input tokens | Users | Ordinary tokens/s | Candidate tokens/s | Ordinary p95 first output (s) | Candidate p95 first output (s) |
| --- | --- | --- | --- | --- | --- |
| 128 | 1 | 138.81 | 138.93 | 0.027 | 0.027 |
| 128 | 4 | 194.76 | 390.53 | 0.109 | 0.110 |
| 512 | 1 | 131.55 | 131.68 | 0.078 | 0.076 |
| 512 | 4 | 180.83 | 338.57 | 0.323 | 0.306 |
| 2048 | 1 | 87.29 | 88.15 | 0.592 | 0.561 |
| 2048 | 4 | 106.82 | 148.52 | 2.363 | 2.240 |

Four-user complete-round throughput improves by 2.01x, 1.87x and 1.39x;
single-user throughput remains within about 1%. Every measured output matches
exactly across variants and repeats. Independent quantized CPU-reference
replays pass all 84 checks on both
[0.8B](evidence/amd/qwen35-q8-hip-small-batch-replay4096.json) and
[2B](evidence/amd/qwen35-2b-q8-hip-small-batch-replay4096.json).
This is a measured serving improvement at the diagnostic 4K tier. It does not
establish 32K support, a maximum-user SLO result or cross-engine parity.
The [candidate lifecycle gate](evidence/amd/qwen35-q8-hip-small-batch-service4096.json)
passes 20 alternating swap cycles per one/two-resident-model configuration,
streaming, concurrent requests, cancellation, load-failure recovery and disk
reload. Host cache reaches zero; device allocation remains 32 MiB across
unload/reload/unload in both configurations. Ordinary-entry-point integration
validation is tracked separately from this isolated experiment.


The runtime now selects the measured matvec policy only for FP16 Q8 projections
with two through four rows on `gfx1151`. Other architectures and row counts
retain their dispatch. The ordinary entry point passes independent 84-check
replays on [0.8B](evidence/amd/qwen35-q8-hip-default-small-batch-replay4096.json)
and [2B](evidence/amd/qwen35-2b-q8-hip-default-small-batch-replay4096.json), plus
a [normal CLI lifecycle integration check](evidence/amd/qwen35-q8-hip-default-small-batch-service4096.json)
with two swap cycles per residency configuration and stable 32 MiB release.
The longer 20-cycle candidate proof above exercises the same selected kernel.
Reproduce the original A/B control at its recorded revision; the current
ordinary policy already includes the improvement.

## Bounded hybrid prompt chunks

The ROCm Triton scan text path supports opt-in `--prefill-chunk-size` through
the existing scheduler. It retains convolution/recurrent state across chunks,
writes full-attention KV at the correct offsets and uses lower-right causal
attention for continuation chunks. Partial chunks do not produce output logits;
a new prompt at offset zero resets recurrent history. Prefix snapshots remain
explicitly unsupported for hybrid models because KV-only snapshots omit state.
Vision, FLA, other vendors and fallback paths are not qualified by this change.

Independent 4K CPU-reference replays pass all 84 checks for both
[0.8B with 127-token chunks](evidence/amd/qwen35-q8-hip-chunk127-replay4096.json)
and [2B with 257-token chunks](evidence/amd/qwen35-2b-q8-hip-chunk257-replay4096.json).
The [ordinary full-prompt regression](evidence/amd/qwen35-q8-hip-chunk-change-full-replay4096.json)
also passes all 84 checks. A separate real-device
[boundary/isolation control](evidence/amd/qwen35-q8-hip-chunk-boundaries512.json)
passes 52 comparisons, including 1/2/3-token chunks, active decode between
chunks and reuse after abandoning a partial prefix. That control compares
against dllm full-prompt execution; it complements the independent replays.
Invalid continuation offsets are rejected without changing caches or lengths.

The [real CLI lifecycle check with 31-token chunks](evidence/amd/qwen35-q8-hip-chunk31-service4096.json)
passes both residency capacities, switching, streaming, concurrency, cancellation,
load recovery and reload, with host cache zero and stable 32 MiB device release.
These remain small-model development gates. They do not establish 32K context,
long-prompt interference latency or maximum sustainable concurrency.

## First actual 32K capacity and interference gate

A [CPU FP32 quantized reference trace](evidence/amd/qwen35-q8-cpu-fp32-golden32768.json)
now includes a 32,752-token prompt at 32,768 capacity. The
[HIP replay with 512-token chunks](evidence/amd/qwen35-q8-hip-chunk512-replay32768.json)
passes all 84 independent checks, with worst relative RMS 0.002812 and matching
greedy choices. Reference generation alone is not a runtime pass.

The real HTTP fixture separately uses exactly 32,256 prompt tokens plus 512
output tokens, with a 128-token/512-output short stream already active before
the long request arrives. It warms the long and short controls, then repeats
three times. It retains raw IDs, text, SSE timestamps, request usage, environment
and revisions. A one-second maximum visible short-stream gap is declared before
running; this is a coarse interference gate, not the tighter multi-user SLO.

The [512-token-chunk result fails that latency gate](evidence/amd/qwen35-q8-hip-http32768-chunk512-summary.json):
all requests finish with exact token counts and control text, and the short
stream advances during long prefill, but maximum visible gaps are
1.291/1.170/1.188 seconds. Long first-output latency is 37.25/34.92/34.78 seconds;
short-stream p95 gaps are 0.673/0.622/0.615 seconds. Raw reports and logs are
retained as compressed siblings. Stalls are largest at late-prefix chunks.

This establishes a small-model numerical/capacity prerequisite and exposes a
responsiveness gap. It does not qualify practical Qwen3.8+ models, long-document
quality, four simultaneous near-limit prompts, maximum concurrent users, or
larger context tiers. Track those separately in #55/#59/#60.

The one controlled [256-token-chunk follow-up](evidence/amd/qwen35-q8-hip-http32768-chunk256-summary.json)
passes the same one-second ceiling in all three rounds, with exact text/counts
and 126 short-stream progress chunks during long prefill. Maximum gaps are
0.552/0.603/0.603 seconds; p95 gaps remain 0.440/0.471/0.475 seconds, and long
first output remains 35.92/38.26/38.19 seconds. This mitigates the coarse stall
gate without meeting the tighter responsiveness target. No chunk-size default
has been changed. Investigate measured attention dispatch/cost before another
chunk-size sweep.

## Isolated attention backend investigation

The [forced-backend probe](evidence/amd/qwen35-hip-sdpa-backends.json) uses actual
0.8B attention geometry, identical synthetic FP16 inputs, CPU FP32 math
references, profiler operator names and five ABBA timing cycles. The process
explicitly enables `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; no host-wide
setting or runtime default is changed. PyTorch's
[upstream platform test gate](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_cuda.py)
identifies gfx1151 under this experimental flag; local proof is still required.

All eight numerical comparisons pass (relative RMS <0.001, max absolute <0.01).
At 512 query tokens over 32K KV, observed math versus Flash SDPA takes
223.24 versus 31.55 ms, with 1,937,773,056 versus 2,115,072 additional peak bytes.
At 256 queries over 32K, the times are 113.49 versus 13.38 ms. These are
attention-operation results, not whole-model serving speed. They also expose
why a fixed 512 MiB prefill workspace allowance does not cover the math path
at every long-context shape; #60 retains full workspace admission as a gate.

The [complete 0.8B near-32K model replay with this flag](evidence/amd/qwen35-q8-hip-flash-chunk256-replay32768.json)
passes all 84 independent checks (worst relative RMS 0.002367). Serving,
lifecycle and wider geometry remain separate gates.

The [same 32K HTTP fixture using the experimental attention path](evidence/amd/qwen35-q8-hip-flash-http32768-chunk256-summary.json)
passes all three rounds with exact output/counts and 126 short-stream progress
chunks during long prefill. Long first output drops to 10.250/10.252/10.238
seconds, maximum short-stream gaps to 0.126/0.126/0.125 seconds, and p95 gaps to
0.106/0.106/0.105 seconds. These sequential engineering runs are not a full
ABBA serving comparison, but preserve the same checkpoint, storage precision,
context, prompts, output count and chunk size. They still miss the provisional
100 ms p95 responsiveness target; practical-model quality and user saturation
remain open.

ROCm startup now enables the qualified gfx1151 attention path within the dllm
process before requesting GPU execution. The existing gfx1151 admission guard
remains in force. An explicit `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` retains
the diagnostic fallback; other values are rejected. This changes no host shell
configuration or existing service. Runtime status reports the resolved setting.
The [automatic-startup 32K replay](evidence/amd/qwen35-q8-hip-auto-flash-replay32768.json)
passes all 84 independent checks without manually supplying the flag. The
[normal-CLI 20-cycle lifecycle proof](evidence/amd/qwen35-q8-hip-auto-flash-service4096.json)
passes both residency configurations, concurrency, streaming, cancellation,
load recovery and reload, with zero tracked tensor allocation and zero host
cache after unload/reload/unload. Driver/context memory is not measured by that
allocator counter. The launcher may show an unset flag while the child server's
reported resolved setting is `1`.

## Practical Qwen3.8-27B capacity preflight

The pinned [checkpoint inventory](evidence/amd/qwen38-27b-inventory.json) and
[shape-only capacity calculation](evidence/amd/qwen38-27b-capacity.json) use
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The actual hybrid text
weight mapper accepts all required tensor shapes, leaving only vision/MTP
weights outside this text path. This is not weight-value or inference proof.

Packed Q8 text weights require 29,772,773,376 bytes (27.73 GiB), with
2,302,410,752 bytes (2.14 GiB) of persistent state per 32K slot. Four such slots
plus weights require about 36.31 GiB before workspace, allocator overhead,
graphs, host copies and loading peaks. HOMER is the first single-device target;
these Q8 weights alone exceed the A30's memory.

Q4 storage is calculated only as a hypothetical capacity target: the hybrid
loader currently rejects Q4. A 15.80 GiB packed-storage estimate does not mean
Q4 execution or quality is supported. The original checkpoint download is
55,563,006,776 bytes, including vision/MTP tensors; the download preflight
retains 64 GiB of free disk without removing existing caches.

Reproduce the shape calculation without model data:

```sh
python -m scripts.experiments.plan_hybrid_capacity \
  docs/evidence/amd/qwen38-27b-inventory.json \
  docs/evidence/amd/qwen38-27b-safetensors-metadata.json.gz \
  /tmp/qwen38-27b-capacity.json
```

The first independent text-reference load of the real checkpoint exposed a
[reference-only quantization failure](evidence/amd/qwen38-reference-vision-failure.log.gz):
it attempted to pack an unused vision projection with input width 4304, which
is not divisible by the Q8 block width. The reference loader now leaves the
vision/MTP branches unchanged for this text-only packing path. This does not
qualify multimodal quantization. A regression test retains that non-packable
vision projection while checking the text projection against upstream packing.
The completed [download record](evidence/amd/qwen38-27b-download.json) retains
the exact snapshot and remaining disk space.

The first real Qwen3.8-27B Q8 replay at 512 context **fails** the numerical gate:
[chunk-127 replay](evidence/amd/qwen38-27b-q8-hip-chunk127-replay512.json)
passes 68/84 checks; the [unchunked control](evidence/amd/qwen38-27b-q8-hip-full-replay512.json)
passes 70/84. All 84 greedy choices match in both runs, but maximum relative
RMS errors are 0.214906 and 0.103750 respectively. These are failures, not a
support or performance qualification. Actual packed host weights match the
29,772,773,376-byte preflight estimate.

The [independent reference](evidence/amd/qwen38-27b-q8-cpu-fp16-golden512.json)
uses CPU FP16; [run provenance](evidence/amd/qwen38-27b-reference512-run.json)
records source hashes and resource limits. The unchunked failure means chunking
alone does not explain the drift. A separate FP32 reference is the next
precision control; tolerances remain unchanged.

`verify_stress.py --reference-only --reference-device cpu --reference-dtype
float32 --reference-quantization q8_0 --packed-cpu-reference` provides that
control without full-model FP32 weight residency. It accepts only all-BF16
Qwen3.5-family source tensors, checked before loading; the BF16 load therefore
preserves checkpoint values. Upstream gguf-py packs each text Linear and
expands it for individual CPU FP32 operations. Transformers retains all model
execution; no dllm model math or quantization kernels enter this reference.
Vision/MTP remain outside the text packing path. This validation-only mode
rejects non-CPU or non-FP32 use and is not a hybrid-serving implementation.

The full packed-storage CPU FP32 control hit its 1,800-second bound without a
completed trace; [timeout evidence](evidence/amd/qwen38-27b-fp32-reference512-timeout.json)
and the [log](evidence/amd/qwen38-27b-fp32-reference512-timeout.log.gz) are
retained. The last sampled scope peak was 62,822,502,400 bytes. This is neither
a numerical pass nor a model failure. Follow-up diagnostics use
`reference_fixed_trace.py` to preserve the original token sequence for one
failing slot and a declared final source event. Partial progress is retained
separately, and the replay tool rejects traces explicitly marked incomplete.
