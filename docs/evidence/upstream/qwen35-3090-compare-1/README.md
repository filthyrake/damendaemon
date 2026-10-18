# Qwen3.5-0.8B comparable upstream HTTP benchmark — vLLM vs damendaemon (#1, 2026-10-18)

## Result

The one remaining #1 gate — "comparable upstream HTTP performance gate: vLLM
startup attempts failed" — is resolved. vLLM 0.32.0 starts and serves the
identical Qwen3.5-0.8B safetensors checkpoint on the desktop RTX 3090, and both
stacks were measured with the same runner on the same device:

| Stack | Mean latency (16 req) | Median | Aggregate throughput |
|---|---|---|---|
| vLLM 0.32.0 (CUDA, compiled) | 0.1171 s | 0.1140 s | 306.5 tok/s |
| damendaemon (torch 2.13.0+cu130, triton-hybrid) | 0.4836 s | 0.3832 s | 74.2 tok/s |

vLLM is ~4.1x faster on this workload. That is the honest comparison record;
damendaemon's hybrid-eager serving path is not competitive with vLLM's compiled
continuous batching on short-prompt small-batch requests, and the earlier
within-damendaemon timings (~300 tok/s decode self-timing) measured a different
thing (steady decode gaps, not end-to-end request latency).

## Conditions

- Checkpoint: Qwen3.5-0.8B snapshot `2fc0636...` (safetensors, unquantized),
  same path served by both stacks. Tokenizer handling matches: identical
  prompt-token counts per prompt on both servers.
- Device: rocky10-desktop RTX 3090 (24 GB), idle, one visible GPU
  (`CUDA_VISIBLE_DEVICES=GPU-24ef3d4c...`); servers run sequentially.
- Workload: `scripts/experiments/upstream_compare_bench.py` — 8 prompts x 2
  rounds sequential greedy requests, max_tokens 64, OpenAI
  `/v1/completions`, per-request wall latency and usage-declared completion
  tokens (574 tokens total on both stacks).
- vLLM: port 8128, `--gpu-memory-utilization 0.5 --max-model-len 4096`, torch
  compile active. damendaemon: `serve --device cuda --context 4096
  --batch-size 4 --no-graph`, triton-hybrid backend.

## Scope

Comparison evidence for #1's upstream-gate acceptance only. This is not a
matched-performance scorecard qualification (#54), not an SLO/concurrency gate
(#59/#63), and not a claim that either number transfers to other GPUs,
contexts, batch shapes or quantized checkpoints. Desktop wall times; no
thermal isolation. Damendaemon-side optimization (compiled batching, fused
decode) remains future work under the performance issues.

## Files

- `bench-vllm.json` — vLLM run (16 requests, per-request latency/usage)
- `bench-damendaemon.json` — damendaemon run (identical workload)
