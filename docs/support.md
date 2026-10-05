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
| Llama / SentencePiece GGUF | Published TinyLlama 1.1B Chat Q4_K_M | NVIDIA A2 | Model/layout/serving and model swapping through 2048 context |
| Llama 3.2 | 1B Instruct mirror | NVIDIA A2, Apple M4 | Scaled RoPE; tested contexts differ by device |
| Gemma 3 text | 270M IT mirror | NVIDIA A2, Apple M4 | Text architecture only; separate precision policy |
| Phi-3 | Mini 4K | NVIDIA A2; limited Apple slot proof | No LongRoPE or interleaved prefill |
| Mistral | 7B Instruct v0.1 | NVIDIA A2 | Sliding-window boundaries; Q8 and Q4 proof |
| StarCoder2 | 3B | NVIDIA A2, Apple M4 | FP32 path; Q8 separately qualified on CUDA |
| OLMoE | 1B-7B 0924 Instruct | NVIDIA A2 | Q8 weights; expert prefill performance remains limited |
| GPT-OSS | 20B, native MXFP4 | NVIDIA A2 | Text Harmony chat + raw completions; auto/none function calls and results at 1024 context/two slots; forced tools, 120B and long contexts pending |
| GPT-2 | Original GPT-2 | NVIDIA A2, Apple M4 | Learned positions; FP32, raw completions |
| GPT-NeoX | Pythia-160M | NVIDIA A2, Apple M4 | Partial rotary / parallel residual; FP32 |
| BLOOM | BLOOM-560m | NVIDIA A2, Apple M4 | ALiBi; FP32, 2048 context |
| T5 / FLAN-T5 | Small checkpoints | NVIDIA A2, Apple M4 | FP32 encoder-decoder raw generation; no quantization or prefix reuse |
| Mamba | 130M, 370M | NVIDIA A2, Apple M4 | FP32/Q8 raw generation; recurrent-state prefix snapshots |
| Mamba-2 | 130M | NVIDIA A2, Apple M4 | Single-group configuration; optional qualified CUDA SSD/compensated projection paths |
| BERT cross-encoder | MS MARCO MiniLM-L6-v2 | NVIDIA A2, Apple M4 | FP32 paired query/document relevance; one trained scalar head;512 tokens |
| XLM-RoBERTa cross-encoder | BGE reranker v2 M3 | NVIDIA A2, Apple M4 | FP32 multilingual scalar relevance; reference-tested through2048 tokens; full8192 limit not GPU-qualified |
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
| Model lifecycle | Download pinned snapshots or exact GGUF files; register/load/unload aliases; GPU and host residency; request-selected switching |
| Resident projection cache | Opt-in, admission-accounted FP16 copies for qualified TinyLlama and Qwen3-1.7B K-quants; respectively 1973 and 3281.5 MiB extra VRAM, ordinary packed fallback when budget/free memory is insufficient |
| Persistent prepared weights | Opt-in safetensors cache for dense Q4/Q8; source identity and layout checks; no automatic disk eviction |
| HTTP generation | Text chat and raw text/token-ID completions; streaming SSE; usage and timing; bounded admission and cancellation |
| Sampling | Greedy or request-seeded temperature/top-p/top-k; repetition, frequency and presence penalties; token bias; text stops |
| Token scores | Opt-in chat selected-token logprobs and up to 20 alternatives; raw distribution; no custom text stops with scores |
| Structured output | Optional constrained JSON / bounded JSON schema including anyOf, with explicit unsupported combinations |
| Function tools | Qualified Qwen2/Qwen3 JSON calls and result history; auto/none and required/named single-call schemas; strict auto unsupported |
| Direct choices | Unquantized Qwen2/Qwen3 candidate-token logits/probabilities without answer generation; explicit raw prompts, no calibrated confidence |
| Batching | Continuous decode, up to 16 slots; CUDA graph replay; opt-in bounded CUDA dense paged KV with admission reservations |
| Prompt scheduling | Opt-in interleaved prefill for qualified dense architectures; bounded request-local state; optional 0–10 ms idle prompt coalescing (default 0), with measured first-output tradeoffs |
| Prompt graph reuse | Opt-in bounded Qwen2/Qwen3 CUDA single-prefill and candidate-scoring graphs; cold capture cost, exact shape/slot/choice keys |
| Speculative decoding | Opt-in Qwen2/Qwen3 dense CUDA greedy n-gram proposals with full-target verification; one active request, ordinary-path fallbacks |
| Prefix reuse | Opt-in bounded GPU prefix snapshots for qualified dense architectures; model/precision isolation |
| Reranking | Bounded query/document lists, raw relevance scores, stable top_n ranking and optional returned text; no silent truncation |
| Embeddings | Text/token-ID inputs, float/base64 output for the qualified BERT sentence encoder |
| Weight formats | Hugging Face safetensors; standalone Qwen2/Qwen3/Llama 3 BPE GGUF F16/F32/Q4_0/Q8_0; Llama-style SentencePiece BPE including native positional score ties (Mistral Q4 and published TinyLlama Q4_K_M model/layout/serving proof through 2048 context); experimental CUDA Q4_K/Q5_K/Q6_K with mixed fused projections; own on-load Q8_0/Q4_0 conversion where supported; checkpoint-native GPT-OSS MXFP4 |
| Chat templates | Checkpoint defaults or explicit per-model sandboxed UTF-8 overrides; stable through eviction/reload |
| Position schemes | Standard/linear/Llama3 RoPE, fixture-qualified static YaRN; learned positions; partial rotary; ALiBi |

The runtime does **not** currently provide general vLLM/llama.cpp parity.
Missing major capabilities include paged KV for other architectures/devices, shared prefix pages, KV quantization,
multi-GPU tensor/pipeline parallelism, adapters, broader GGUF architectures and remaining K/IQ formats, AWQ/GPTQ checkpoint import,
multimodal inference, additional tool-call formats, prompt/raw-completion token log-probabilities,
trained decision-head checkpoints, and a CPU inference backend. Model registration is
process-local. The HTTP API is a supported subset, not full OpenAI or Ollama API
compatibility. The development server is loopback-only and is not an authenticated
multi-tenant service.

K-quant qualification covers Llama 3.2 1B and Qwen3-0.6B Q4_K_M at 512
context, with Qwen3 independent model checks also passing at 2048.
Qwen3-1.7B's pinned Q4_K_M conversion additionally passes independent model
checks at 2048. Its optional, admission-accounted resident cache passes
independent extended model, layout, server and memory-release checks through
2048 with the standard Qwen3 precision policy; a 4096 MiB budget admits
3281.5 MiB of projection copies. Other Qwen sizes and the FP32 mode are excluded.
Published TinyLlama Q4_K_M passes independent model and fixed/paged/batched/chunked layout
checks through 2048, with normal HTTP and model swapping also qualified at 2048. The optional
FP16 residual-int8 backend permits Q4/Q6 single-request decode through context
2048 for the qualified TinyLlama topology, while other models retain the 512
limit. Q5_K_M and longer Llama 3.2 contexts retain documented mixed/FP16 numerical gaps.
The explicit Llama FP32 mode passes independent Q4_K_M/Q5_K_M model checks at
2048 context. Its separate optional FP32 residual CUDA backend accelerates
one through four rows. Separate opt-in compensated FP32 prefill targets prompts
of at least 128 rows in the same Llama scope. Fixed/paged/graph/chunk and server/lifecycle proof covers
512 context. These checkpoint-specific results do not establish qualification
for every model using these block formats.

Performance is workload-specific. Comparable results include losses as well as
wins; a passing numerical gate is not a claim of throughput parity. In particular,
Mistral Q4 single-request decoding still trails llama.cpp, and several concurrent
workloads still trail vLLM. See the raw trials accompanying each comparison.

Uniform window metadata in Llama-layout GGUF files maps to the Mistral decoder.
The pinned Mistral 7B Q4 fixture passes independent and layout checks at context
4104 across its 4096-token window, plus real streaming, switching, cancellation
and reload checks against exact source-decoded traces. Mixed local/global patterns are rejected.
This implements attention semantics with full-context KV storage, not a rolling
window allocation. Its GGUF chat tokenizer matches the native format reference
and can differ from the HF fast tokenizer even for the same rendered template.

Recent evidence: [GGUF import and templates](evidence/gguf/), [T5](evidence/t5/), [Mamba](evidence/mamba/),
[Mamba-2](evidence/mamba2/), [function tools](evidence/tools/),
[direct token choices](evidence/scoring/), [trained reranking](evidence/reranking/),
[speculative decoding](evidence/speculation/) and [upstream research](upstream-inference-research.md).
