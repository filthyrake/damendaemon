# damendaemon documentation

[Project overview](../README.md) · [Support matrix](support.md) · [Parity milestones](https://github.com/filthyrake/damendaemon/issues/3)

## Use damendaemon

| Guide | What it covers |
| --- | --- |
| [Getting started](getting-started.md) | CUDA installation, model downloads, immutable revisions, Apple Silicon |
| [Serving and model management](serving.md) | HTTP requests, streaming, image input, aliases, residency, JSON and tools |
| [Quantization and weight caches](quantization.md) | Per-alias precision, Q4/Q8, AWQ, GGUF, optional CUDA libraries |
| [Performance tuning](performance.md) | Chunked prefill, prefix reuse, paged KV, speculation, prompt coalescing |
| [Multi-GPU execution](multi-gpu.md) | Qualified layer partitions, fast swaps, network and recovery limits |
| [Model-specific guidance](models.md) | Embeddings, reranking, text decoders, recurrent models and MoE |
| [Benchmarking](benchmarking.md) | Repeatable HTTP measurements, reported metrics, CUDA hardware probe |

Examples assume commands run from the repository root after installation.
Model acceptance does not imply qualification at every context or precision.

## Understand the implementation

- [Product direction: modern models, long context and concurrent users](direction.md)
- [Architecture and residency design](architecture.md)
- [Current support boundaries](support.md)
- [AMD/ROCm target and current qualification status](amd.md)
- [Fixed performance scorecard acceptance](https://github.com/filthyrake/damendaemon/issues/54)
- [Upstream inference research](upstream-inference-research.md)

## Inspect qualification evidence

- [Architecture correctness and serving qualification](architecture-parity.md)
- [Model matrix and longer contexts](model-matrix.md)
- [Fused kernels and batching](batching-proof.md)
- [Hybrid models and Apple Silicon](hybrid-and-mps.md)
- [Vision qualification](vision-qualification.md)
- [Multi-GPU execution, failure recovery and swap measurements](evidence/multi-gpu/README.md)
- [GGUF numerical gates and performance](evidence/gguf/README.md)
- [Metal quantization](evidence/quantization/metal/README.md)
- [Original runtime baseline](runtime-proof.md) and [A2 hardware probe](a2-hardware.md)

Evidence reports preserve the measured scope and historical results; they do not
replace the [GitHub issue tracker](https://github.com/filthyrake/damendaemon/issues).
