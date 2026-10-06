# Product direction

[Documentation](README.md) · [Roadmap and ordered work](https://github.com/filthyrake/dllm/issues/3)

The goal is independent inference for large modern models, with as many users
as possible receiving useful token speed and responsive service. Fast model
switching remains part of the product. This page records the user's north stars
from 2026-10-06; it does not describe current support. GitHub Issues own the work
and acceptance gates.

## Models

| Target | Work and qualification |
| --- | --- |
| Qwen3.8 and newer | [#55](https://github.com/filthyrake/dllm/issues/55); first concrete target Qwen3.8-27B |
| GLM-5.3 | [#56](https://github.com/filthyrake/dllm/issues/56) |
| GLM-5.3-Flash | [#57](https://github.com/filthyrake/dllm/issues/57), including its separate multimodal/hybrid architecture |
| DeepSeek V4 Flash | [#58](https://github.com/filthyrake/dllm/issues/58) |

The [official configuration inspection pins](evidence/planning/model-targets-2026-10-06.json)
record the source URLs and revisions without downloading model weights.
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) identifies its text
architecture as `qwen3_5_text`, making the current hybrid work relevant.
[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), and
[DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) have
separate sparse/hybrid/compressed-attention and expert-routing requirements.
Recognizing those configurations is not full-model qualification.

Qwen3.5-0.8B/2B and small Qwen3 models remain development controls. They do not
replace the practical targets. Older-model implementation and tuning are
deferred; their existing evidence and cheap regressions remain available.

## Context and concurrency together

Minimum usable context is **32,768 tokens**. Preferred tiers are **65,536** and
**131,072**, with **262,144+** the long-term target where the checkpoint supports
it. Capacity includes prompt plus reserved output. Large configuration values
or short prompts on a large cache do not establish long-context support.

The primary performance result is **maximum sustainable concurrent users within
a declared speed and latency budget**. Report per-user output speed, p95/p99
first-token latency and inter-token gaps, queue delay, errors, rejected work,
fairness, memory and aggregate goodput. Queued users are not actively served
users, and high aggregate throughput does not excuse individual users stalling.

The initial engineering profiles measure 10/20/40 output tokens/s per user,
with 20 tokens/s a provisional interactive target. Short-prompt p95 first-token
latency <=2 seconds and p95 inter-token gap <=100 ms are starting measurement
budgets, not user-promised SLAs. Longer-context prefill budgets must be declared
for each hardware/model tier before measurement. Sweep concurrency
1/4/8/16/32/64/128 until a measured resource or latency limit, keeping failing
points visible. [Concurrency #59](https://github.com/filthyrake/dllm/issues/59)
and [context #60](https://github.com/filthyrake/dllm/issues/60) own the details.

This requires admission accounting for weights, KV, recurrent/compressed state,
prefix caches, graph buffers, prefill workspace and communication buffers.
Long prompts must coexist with ongoing decoding. Quality checks include
multi-position retrieval with distractors, composition across documents, code
and summarization, alongside independent numerical/state checks. No silent
truncation or relaxed quality thresholds to make a throughput result pass.

Current 4K benchmarks are diagnostic baselines. They do not satisfy this
context requirement or establish large-model multi-user readiness.

## Hardware and sharding

First-class vendor targets are NVIDIA, AMD, Intel and Apple Silicon, spanning
consumer and datacenter systems. Existing development hardware is A2, A30,
RTX 3090, Intel Arc B580, HOMER's Radeon 8060S/gfx1151 and Apple M4. Additional explicit targets:

- [GB200 and DGX Spark #61](https://github.com/filthyrake/dllm/issues/61).
- [AMD MI300 family #62](https://github.com/filthyrake/dllm/issues/62), separate from HOMER qualification.
- [Intel Arc B580 #63](https://github.com/filthyrake/dllm/issues/63).

**Hardware priority:** qualify the available HOMER, A30, RTX 3090, A2, M4 and B580
first. DGX Spark, GB200 and MI300 follow satisfactory relevant workflows and
performance gates on that tier. The B580 shares the accessible RTX 3090 desktop
and drives the user’s display. Inventory its current free memory before testing;
reserve desktop headroom and use conservative admission limits. Access is
confirmed by the user; Intel runtime support remains unqualified.
Do not block available-machine progress on inaccessible hardware or start new
hardware provisioning merely because its issue exists.

Exact device, CPU architecture, memory topology, compiler/runtime and transport
capabilities need separate real-hardware qualification. Access to new systems
is not assumed. A successful kernel on one vendor/device does not establish
support on another.

[Distributed execution #51](https://github.com/filthyrake/dllm/issues/51) covers
large models sharded across local and network GPUs. Tensor, pipeline and expert
parallelism address different communication patterns; replicas address user
throughput. Capacity gains and speed gains must be reported separately. RDMA
requires real compatible endpoints and transfer proof. Cross-vendor fallbacks
must disclose host staging and communication cost. Peer loss, partial admission,
cancellation and model switching must release all participating resources.

Multi-user service also needs session isolation, quotas, bounded queues,
authentication for non-loopback access, observability and graceful overload.
Those are product requirements, not permission to expose the current server.

## Execution discipline

Finish the current bounded experiment, preserve its evidence, then select the
next failed acceptance gate. Build shared capabilities that advance these
named targets. Do not run a full hardware/model Cartesian sweep or turn a
negative benchmark into an endless chain of unrelated kernel experiments.
Keep one GPU experiment active, preserve existing workloads, and commit/push
verified increments. The [roadmap](https://github.com/filthyrake/dllm/issues/3)
contains the ordered milestones; this document is not a second task tracker.
