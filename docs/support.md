# Current supported scope

This page describes implemented behavior. [GitHub Issues](https://github.com/filthyrake/dllm/issues)
track remaining work; numerical and performance reports are linked below.
Configuration acceptance alone is not checkpoint qualification.

## Qualified model families

| Family | Checkpoints exercised | GPU validation | Important boundary |
|---|---|---|---|
| Qwen2 / Llama-style dense | Qwen2.5, SmolLM2 | NVIDIA A2, Apple M4 | Exact tested sizes/revisions in the evidence |
| Qwen3 dense | 0.6B, 1.7B | NVIDIA A2, Apple M4 | CUDA accurate FP32 arithmetic policy is required for the stricter long-context proof |
| Qwen3.5 hybrid | 0.8B, 2B text backbone | NVIDIA A2; Apple path qualified separately | Dense text only; no vision or hybrid MoE |
| Llama 3.2 | 1B Instruct mirror | NVIDIA A2, Apple M4 | Scaled RoPE; tested contexts differ by device |
| Gemma 3 text | 270M IT mirror | NVIDIA A2, Apple M4 | Text architecture only; separate precision policy |
| Phi-3 | Mini 4K | NVIDIA A2; limited Apple slot proof | No LongRoPE or interleaved prefill |
| Mistral | 7B Instruct v0.1 | NVIDIA A2 | Sliding-window boundaries; Q8 and Q4 proof |
| StarCoder2 | 3B | NVIDIA A2, Apple M4 | FP32 path; Q8 separately qualified on CUDA |
| OLMoE | 1B-7B 0924 Instruct | NVIDIA A2 | Q8 weights; expert prefill performance remains limited |
| GPT-2 | Original GPT-2 | NVIDIA A2, Apple M4 | Learned positions; FP32, raw completions |
| GPT-NeoX | Pythia-160M | NVIDIA A2, Apple M4 | Partial rotary / parallel residual; FP32 |
| BLOOM | BLOOM-560m | NVIDIA A2, Apple M4 | ALiBi; FP32, 2048 context |
| BERT sentence encoder | all-MiniLM-L6-v2 | NVIDIA A2, Apple M4 | Specified mean-pooling/L2 pipeline; 384 dimensions, 256 input tokens |

Exact pins, precision, context, correctness gates and comparisons:
[initial dense matrix](model-matrix.md), [hybrid and MPS](hybrid-and-mps.md),
[additional architectures](architecture-parity.md). These reports preserve
historical measurements; later sections explicitly identify superseding results.
Model licenses remain those of their respective publishers.

## Serving and execution

| Capability | Current behavior |
|---|---|
| GPU execution | Independent forward passes; CUDA/Triton/native optional CUDA C++; Apple MPS/Metal |
| Model lifecycle | Download pinned snapshots; register/load/unload aliases; GPU and host residency; request-selected switching |
| Persistent prepared weights | Opt-in safetensors cache for dense Q4/Q8; source identity and layout checks; no automatic disk eviction |
| HTTP generation | Text chat and raw text/token-ID completions; streaming SSE; usage and timing; bounded admission and cancellation |
| Sampling | Greedy or request-seeded temperature/top-p/top-k; repetition, frequency and presence penalties; token bias; text stops |
| Structured output | Optional constrained JSON / limited JSON schema, with explicit unsupported combinations |
| Batching | Continuous decode, up to 16 slots; CUDA graph replay; opt-in bounded CUDA dense paged KV with admission reservations |
| Prompt scheduling | Opt-in interleaved prefill for qualified dense architectures; bounded request-local state |
| Prefix reuse | Opt-in bounded GPU prefix snapshots for qualified dense architectures; model/precision isolation |
| Embeddings | Text/token-ID inputs, float/base64 output for the qualified BERT sentence encoder |
| Weight formats | Hugging Face safetensors; own on-load Q8_0/Q4_0 conversion where supported |
| Position schemes | Standard/linear/Llama3 RoPE, fixture-qualified static YaRN; learned positions; partial rotary; ALiBi |

The runtime does **not** currently provide general vLLM/llama.cpp parity.
Missing major capabilities include paged KV for other architectures/devices and its combination with prefix snapshots, KV quantization,
multi-GPU tensor/pipeline parallelism, adapters, GGUF/AWQ/GPTQ checkpoint import,
multimodal inference, tool-call protocol handling, token log-probabilities,
encoder-decoder generation, and a CPU inference backend. Model registration is
process-local. The HTTP API is a supported subset, not full OpenAI or Ollama API
compatibility. The development server is loopback-only and is not an authenticated
multi-tenant service.

Performance is workload-specific. Comparable results include losses as well as
wins; a passing numerical gate is not a claim of throughput parity. In particular,
Mistral Q4 single-request decoding still trails llama.cpp, and several concurrent
workloads still trail vLLM. See the raw trials accompanying each comparison.
