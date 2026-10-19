# Candidate 1 — enable CUDA graphs for hybrid decode (--no-graph removal)

Observed: upstream bench on 5d57fe41 (2026-10-18/19) drove both stacks with `damendaemon serve --no-graph` (runtime use_graph=False). For qwen3_5_text HybridExecutor that disables `self.batch_graphs` entirely (hybrid.py:333ff guarded by `if use_graph:`) — decode falls back to eager `forward_batch(...)` + `logits.argmax(-1)` kernel-launch per token plus extra Python dispatch around each layer.

Expected cost vs graph replay captured in BatchExecutor/HybridExecutor.decode_batch:

* graph replay path: 1 kernel-launch amortized, host metadata pushed via pinned staging + single DMA, NCCL-free. Typical per-token saving on 3090 for 0.8B-scale is 3–6 ms in analogous dense models (prior cpu-host evidence: graph ON ~40% throughput lift).
* current eager path pays per-layer launch storm (≈46*small GEMMs+norms) + Python loop remainder counted in 13.48 ms/tok.

Fix: remove `--no-graph` for hybrid comparison runs (default `Runtime(..., use_graph=True)` already, cli `--no-graph` is debug escape hatch, not baseline). No code change to executor required — plumbing already captures graphs (hybrid.py:347 `CUDAGraph(...); logits.argmax(-1)` inside). Ensure `--batch-size 4` capture warms once at start and reuses for batch-1 steady state.

Validation plan on rocky10-desktop (idle 3090, same 8×2 workload):

```
SCENE=Qwen3.5-0.8B-2fc06... damendaemon serve --device cuda --context 4096 --batch-size 4 --port 8128   # no --no-graph
python -m scripts.experiments.upstream_compare_bench --port 8128 --model qwen35 --label damendaemon-graph --output /tmp/graph.json
```

Success predicate (per #534): mean 0.s graph-on ≤ 0.176 s (vLLM 0.117*1.5) or clearly documented delta toward 1.5× if graph alone insufficient. If uplift << 1.5×, revert to documenting remaining scheduling/sample gap as cycle outcome per stopping condition (≤2 cycles before scoreboard doc).
