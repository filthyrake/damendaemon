# Qwen3.6-35B-A3B (qwen3_5_moe) Q8_0 32K concurrency cells — C1/C2/C4 ALL PASS (#668 cell 2 continuation, 2026-11-01)

Direct-executor concurrency cells on the shared A6000
(`GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed`), same pinned snapshot and
quantization as the passing cell-2 Q8 gate (PR #670: `995ad96e…`, q8_0
packed expert projections, Triton kernels, `--no-graph`, context 32768).
The HTTP lifecycle remains cell 3's work; this cell measures executor
capacity and SLO-relevant scaling (#59/#60).

Runner `scripts/experiments/qwen36_35b_q8_concurrency_668.py`
(sha256 `dc9ca4f103c15ddd6429a5580611a3c075523bcfd9a30ad21c014f937243cd47`),
one bounded C{N} attempt per invocation, guarded each time
(`guard_shared_cuda`, thermal 90 °C, reserve 0, timeout 1800 s — the
pre-declared near-capacity envelope on #668; all three guards completed
exit 0, thermal peaks 70/74/75 °C). Prompts: 32,256 tokens/slot —
deterministic per-slot rotations of one corpus (digests in the reports;
slot 0 identical across cells, digest
`cc380b42…` output on C1 = C2 = C4 slot 0). Whole-prefill per slot
(chunked prefill is **not admitted** for CUDA q8_0 fp16 — sequential
per-slot prefill, disclosed), then continuously batched greedy decode
(`decode_batch`) to 512 tokens/slot. Per-slot gates inherited from the
#68 C1 discipline: prefill ≤ 600 s, whole request ≤ 1200 s, exact token
counts, finite logits.

## Results

| cell | prefill s/slot | decode (511 steps) | aggregate tok/s | per-slot tok/s | request s/slot | peak GiB | verdict |
|---|---|---|---|---|---|---|---|
| C1 (1 slot) | 7.81 | 146.7 s | 3.49 | 3.49 | 154.6 | 40.29 | **PASS** |
| C2 (2 slots) | 7.78/7.77 | 176.0 s | 5.82 | 2.91 | 183.8 | 40.98 | **PASS** |
| C4 (4 slots) | 7.76–7.81 | 217.5 s | 9.42 | 2.35 | 225.3 | 42.35 | **PASS** |

- Weights 34.76 GiB (load 115–117 s — no prepared-weight cache for q8
  MoE; the q4_0/qwen3_5_text-only cache does not cover this format,
  noted as follow-up), KV 640 MiB/slot (10 full-attn layers × 2 KV
  heads × 256 dim, fp16, 32K — measured `kv_cache_bytes` matches the
  geometry math exactly), linear states ~2 MiB/slot.
- All gates passed on every slot of all three cells; first 32K prefill
  includes Triton JIT for the large scan shapes (disclosed; the warmup
  at 256 tokens does not cover them — C2/C4 prefills at ~7.8 s show the
  JIT cost is bounded and per-process).
- Peak allocator at C4 42.35 GiB of 44.5 GiB — near-capacity exactly as
  pre-declared; no OOM at any cell.

## The #59/#60 finding — the MoE opens the SLO window the 27B dense could not

Against the qualified 27B dense cells (#68, same card, same discipline):

| | 27B dense Q4 (#68) | 35B-A3B MoE Q8 (this cell) |
|---|---|---|
| 32K prefill / slot | 154.8 s | **7.8 s** (~20×) |
| C2 first output | 308.1 s | **~16 s** (both prefills + first step) |
| C4 first output | 616–625 s (gate FAILURE) | **~32 s** |
| C4 decode aggregate | ~3.2 tok/s (clipped at gate) | **9.42 tok/s** |
| C4 verdict | failed per-request time gates | **all slots pass** |

- **C4 at 32K passes every per-request gate** — 4 concurrent users, each
  32K prompt + 512 greedy tokens, every request ≤ 225 s. The 27B's C4
  was bounded out by the time gates; the MoE re-opens that window.
- Decode scaling is sub-linear but strongly amortizing: 3.49 → 5.82 →
  9.42 aggregate tok/s at 1/2/4 slots (per-slot 3.49 → 2.91 → 2.35).
  The packed-expert dequant-per-step (correctness-first `F.linear`
  semantics, kernel dispatch explicitly deferred per #670) reads the
  full weight set per step regardless of batch, so batched decode
  amortizes weight bandwidth: C4 retains 67% of single-slot per-slot
  speed while serving 4 users (the 27B packed-4 dropped to ~0.8 tok/s/
  slot, ~23% of its C1).
- Prefill is embarrassingly cheap (7.8 s/32K) and shows zero cross-slot
  interference (sequential whole-prefill); concurrency cost is entirely
  in decode.

## Honest limits

- Capacity + SLO feasibility cells with the #68 gate discipline — these
  are **not benchmarks** (shared-card wall-clock, Triton JIT included in
  the first prefill, no warm-loop steady state claimed).
- Greedy continuity only (coherent deterministic outputs, digests
  recorded); no quality, cancellation, reuse or HTTP-lifecycle claim.
- C8 not attempted: 4 slots was the pre-declared ceiling for this cell
  (peak 42.35 GiB leaves no C8 KV headroom on 44.5 GiB); higher
  concurrency needs the Q4-ROCm chain or split-KV work, not this card.
- The decode throughput headroom is kernel work (per-expert slice
  dequant → fused batched dispatch) — explicitly deferred as
  throughput territory in #670 and re-measured here as the scaling
  bottleneck, not silently optimized.

## Artifacts (sha256)

- `c1c.json` `da6363205a51ce4b0cfddd1f86b6e0281611e8575e850d00fb82c71eec620c81`
- `c2.json` `79a7164766f737ff745fd6e61c0227f4726b1e56a6482372021fad2963df85ce`
- `c4.json` `227da253b21493061176f32e6733c249959eaca879c16ea938fad3a1fa1aaae4`
- `guard-c1c.json` `107f3d94f130d1387391344b07acad2d9ac10cfbad590ff6b40ed04eeb2ddd71`
- `guard-c2.json` `51a03da8215030fa486097879b760a9fd01629a880f56544689b49bf0962e4a1`
- `guard-c4.json` `80c583ba1e352b93ff60e2dd08286d59b9a8fd311fb6d8632d9dba334778e227`
- raw logs `c1c.log` / `c2.log` / `c4.log` (mostly empty — the runner is
  incremental-JSON-driven; the reports are the record)

Launch history (retained per discipline): two aborted C1 launch attempts
before the clean run — a local `ssh -f` failure whose remote-side state
was unclear, and a second launch that collided with the first's guard
report path (`FileExistsError`, plus one `ModuleNotFoundError` child
invoked by path instead of `-m`). None touched the GPU (first successful
child started 05:16). The clean sequence is the three cells above.
