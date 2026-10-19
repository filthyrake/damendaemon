# #534 gate 1 profile — 3090 latency decomposition (interim)

Generated 2026-10-19 by `profile_hybrid_decode.py --offline` on macOS (no GPU).

Source benches: `docs/evidence/upstream/qwen35-3090-compare-1/bench-{vllm,damendaemon}.json`

* identical prompts 8×2=16 req, Qwen3.5-0.8B safetensors, RTX 3090 idle, sequential.
* Mean latency 483.6 ms damendaemon vs 117.1 ms vLLM (4.13×). Tokens identical: 574 completions ⇒ 35.9 tok/req avg.

Derived throughput:

| stack | mean | ms/tok |
|---|---|---|
| vllm | 117.1 ms | **3.26 ms/tok** |
| damend | 483.6 ms | **13.48 ms/tok** |

Breakdown still needs live capture — predicted split for PR discussion:

1) graph path: HybridExecutor captures per-batch CUDA graphs when `--use_graph` set (see hybrid.py:333-351) but upstream bench invoked with `--no-graph` on both stacks → decode falls back to `forward_batch` + per-step `argmax` host reads. Expect to reclaim O(ms/tok) once enabled and warmed.
2) sync: each token triggers `next_ids` host copy for scheduler commit; batched pipelined sampling collapses N reads to 1.
3) scheduler: Python per-token loop, logprob struct, stream framing.

Acceptance clause: live `torch.cuda.Event` trace from `profile_hybrid_decode.py --live` on rocky10-desktop is the gate artifact, not these wall-clock divides.

Candidate order proposed: (1) enable/warm CUDA graphs for hybrid decode, (2) eliminate redundant D2H sync, (3) micro-opt scheduler serialization.
