# Fused routed dequant-matvec on the Qwen3.6-35B-A3B MoE (#688)

The `_moe_half` q8_0 branch materialized every active expert slice through
`quantization.dequantize`'s fp32 intermediate (`.float()*scales → .to(fp16)`,
~12 MB per expert slice) before two `F.linear` calls — at C6 decode ≈ 48
active experts × 40 layers of pure dequant work per step, executed as a
per-expert Python loop (~1920 dequant launches + ~3840 `F.linear` launches
per decode step).

This change routes the decode path through the existing production-proven
`moe_kernels.routed_linear`/`combine` (the gpt-oss/olmoe family's surface —
register-resident dequant, the fp32 intermediate never exists; 3 launches
per layer instead of ~144). Admission is structural: single-position rows
(decode steps) with valid launch geometry (`rows*TOP ≤ 65535` — the kernel's
grid-Y limit), int8 (q8_0) stacks, CUDA, kernels import. q4_0
paired-nibble stacks and multi-position prefill keep the materialize path
(q4 has different unpacking semantics; prefill amortizes the dequant
intermediate in cuBLAS GEMM better than per-row matvecs). The materialize
path remains as the structural fallback; `DAMENDAEMON_MOE_FUSED_ROUTED=0`
is the A/B control.

## Measured (A6000, both phases guarded, one invocation each)

Qwen3.6-35B-A3B @ 995ad96e, q8_0 packed experts, context 32768,
**8192-token prompts** (so six sequential prefills + 512-token decode fit
one 1750 s guard window; decode at 8K KV positions is cheaper than the
32K C6 record's attention — the A/B delta is the measurement, the C6
1.94 tok/s/user numbers are context, not the control), 6 slots,
deterministic rotations, 512 greedy tokens/slot, Triton, --no-graph.
Runner: `scripts/experiments/qwen36_35b_fused_routed_ab_a6000.py`.

| phase | prefill (6 slots) | decode | per-slot rate | peak |
|---|---|---|---|---|
| fused (`routed_linear`/`combine`) | 17.5 s | **20.5 s** | **25.03 tok/s** | 40.82 GiB |
| materialize (control) | 17.2 s | 245.5 s | 2.09 tok/s | 40.82 GiB |

**Decode is 12.0× faster per slot (25.03 vs 2.09 tok/s); digests are
byte-identical across all 6 slots** (`b4c2c64072d0796b…`, all six match
across phases — the fused path's fp32 route-sum accumulation vs the
materialize path's fp16 `index_add` never flipped a greedy argmax over
6×512 tokens). Prefill timings are identical by construction (both phases
run the materialize path there — the amendment below).

Honest characterization of the 12×: measured jointly, mechanism = the
launch-count collapse (~5760 kernel launches per decode step → ~120) plus
the elimination of the ~23 GB/step fp32-intermediate traffic. The
decomposition is not separately claimed; the launch count is the
dominant term at this geometry (23 GB at ~760 GB/s ≈ 30 ms of pure
bandwidth vs ~480 ms measured for materialize).

Gate verdict (all declared pre-launch on #688):
1. both phases 6/6 slots, 512 tokens, 0 OOM, finite logits — PASS
2. control sanity ≥ 1.0 tok/s/slot (2.09) — PASS
3. fused decode strictly < materialize (20.5 < 245.5, 12.0×) — PASS
4. fused peak ≤ materialize (40.82 = 40.82) — PASS
5. digests compared and reported, not gated — byte-identical (stronger
   than required)

## First-launch finding (retained, root-caused, fixed pre-measurement)

Launch 1 of the fused phase died on the first 8K prefill with
`Triton Error [CUDA]: invalid argument`: `routed_linear`'s grid is
`(cdiv(outputs, tile), selected.numel())` and CUDA's grid-Y limit is
65535 — 8192 rows × TOP 8 = 65536 exceeds it by one. Warmup (256 rows)
passed, which is why construction and warmup were clean. The fix is the
decode-shape admission above (declared on #688 before relaunch; gate
values unchanged). The fused phase's engagement is proven by the decode
timing itself — a silent fallback would have reproduced the control's
~245 s.

The dispatch contract is CPU-pinned in `tests/test_hybrid_moe.py`:
env validation (garbage refuses), env on/off identical on the fallback
path, packed-CPU executors still refused at construction.

## Artifacts

- `fused_cell.json` / `materialize_cell.json` — full cell reports
- `g688a.json` / `g688b.json` — guard reports (phase passed, 0 exit)
- `runner.py` — the A/B runner verbatim

Shared-host disclosure: A6000 idle before/after; one transient
MemAvailable dip (Chrome/Slack renderer churn) refused a launch
legitimately and was retried after recovery — retained as a finding.
