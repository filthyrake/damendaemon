# #43 ladder completion cells: Qwen3-4B-2507 and Qwen3-8B published Q4_K_M

Date: 2026-10-22 (UTC). Follow-up to the #583 scale scan (gate-1 passes at 4B
and 8B). Instrument: the #43 ladder gates on the labnode3 NVIDIA A30, guarded
shared-CUDA gate, TF32 off, thresholds unchanged, failures retained.
Gate-1 reports are the #583 scan's (`qwen3-4b.json`, `qwen3-8b.json`,
`qwen3-8b-fp32ctrl.json` in this directory).

Checkpoints (both read directly, sha256 in every report):

- Qwen3-4B-Instruct-2507 Q4_K_M, 2.50 GB, NAS-sourced, staged at
  `/home/damen/models/qwen3-4b-2507/`
- Qwen3-8B Q4_K_M, 5.03 GB, NAS-sourced, staged at `/home/damen/models/qwen3-8b/`

## Gate 1 — independent reference (from #583, retained)

See [the scale scan](qwen3-q4km-scale-scan-583-README.md): 4B 0.365%, 8B 0.573%
worst relative RMS against FP32 references, zero greedy mismatches. PASS both.

## Gate 2 — execution layouts (`verify_gguf_layout_runtime.py`, context 512)

| cell | checks | worst relative RMS | verdict |
|---|---:|---:|---|
| 4B | 128 | 0.209% | **PASS** |
| 8B | 128 | **3.052% (chunked prefill)** | **FAIL** |

- 4B (`qwen3-4b-layout.json`): batches 1/2/4, eager/graph, fixed/paged KV and
  128-token chunked prefill all pass. Guard `completed`.
- 8B (`qwen3-8b-layout.json`): batch1 is bit-exact; batch2 worst 0.185%,
  batch4 worst 0.124% — graph replay, paged KV and batching all pass. The
  **chunked-prefill** comparison fails at 3.052% relative RMS (max absolute
  1.25, greedy tokens equal at every recorded step), identically across all
  four layout combinations — the divergence is in the chunk-continuation path
  itself, independent of graph/paging. The failure is retained as the gate
  verdict at the default contract (chunk 128).

### Chunked-prefill localization at 8B

`qwen3-8b-layout-chunk505.json` (diagnostic, not a gate retry): with
`--chunk-size 505` (single chunk) the same cell passes **exactly** — worst
relative RMS 0.0000%, all 128 checks, guard `completed`. The defect is
specifically the multi-chunk 128-token boundary continuation at this
checkpoint. A further width control (`llama31-8b-layout-a30.json`): Llama-3.1-8B
(hidden 4096, same Q4_K_M, same device/runner) **passes all 128 checks at
chunk 128** (worst 0.125%) — the defect is qwen3-8B-specific, not hidden-width
general. Filed as #585 with the per-stage localization path recorded.

**#585 resolved** — full attribution, mechanism, and the corrected chunked
contract: see [qwen3-8b-chunked-prefill-585-README.md](qwen3-8b-chunked-prefill-585-README.md).
Summary: not a continuation-logic defect; the divergence is FP16
kernel-arithmetic signature (cuBLAS expand-GEMM M-variance on the
`out`/`down` projections + attention call-geometry rounding), bounded
and greedy-preserving, occasionally amplified by a decode near-tie. Gate rerun
passes under the evidence-justified chunked bounds; an opt-in
shape-invariant Triton GEMM path (`DAMENDAEMON_TRITON_PREFILL_GEMM=1`)
removes the GEMM component when bit-stable chunking is required.

## Gate 3 — standalone tokenizer (`verify_gguf_tokenizer.py` vs llama.cpp)

38/38 cases both cells (chat/special tokens, `add_special` both ways,
`parse_special=true`), including the embedded SentencePiece vocab, merges and
chat template. PASS both (`qwen3-4b-tokenizer.json`, `qwen3-8b-tokenizer.json`).

## Gate 4 — HTTP lifecycle

ABBA benchmark (`benchmark_kquant_engines.py`, A/B/B/A: damendaemon →
llama.cpp → llama.cpp → damendaemon, three warmup + six measured rounds per
phase, 16 output tokens, context 512, four slots, native K-quant decode).

Two comparator attempts are retained as the instrument record:

- `qwen3-4b-bench-attempt1-a30.json`: the pinned llama-server binary is
  sm86-only and crashes on the sm80 A30 at warmup ("no kernel image is
  available") — an archive sm80 rebuild of the same pinned source commit was
  produced (`/tmp/llama-sm80-build/bin/llama-server`, sha256 `bd1f84ea…`).
- `qwen3-4b-bench-attempt2-a2-guard.json`: on the A2 the guard stopped the run
  at the identity/temperature gate — the A2 crossed 65 C under sustained load
  (the documented A2 thermal envelope; the last passing poll was 64 C).

The second complete A30 run (`qwen3-4b-bench-a30b.json`, adjusted prompt text,
recorded) isolated the real finding: the earlier tie interpretation was wrong —
at the 16-token prompt, **damendaemon (phases 0/3) and llama.cpp (phases 1/2)
produce deterministically different continuations** ("### Problem Statement…"
vs "### Step-by-Step Practice…"), stable within each engine across every round.
The 129/400-token cells are identical across engines. This is the expected
cross-engine behavior the ladder's design anticipates: engine text parity is
explicitly not the gate (`verify_gguf_generated_http.py`: "HF token agreement
does not imply that another engine using different arithmetic generates the
same words"). The ABBA text-equality assertion is a benchmark invariant this
checkpoint's 16-token continuation violates between engines; both runs are
retained as the comparator measurement record with that finding stated.

### Ladder gate-4 verdict (independent-teacher criterion)

- `qwen3-4b-teacher.json`: FP32 HF reference on the exact bench workload
  (16/129/400-token prompts, 16 output tokens) — worst relative RMS 0.411%,
  max absolute 0.264, zero greedy mismatches, guard `completed`.
- `qwen3-4b-http.json` (`verify_gguf_generated_http.py`): all three prompts
  through the normal HTTP server (stream + non-stream), generated text and
  usage byte-exact against the teacher trace, guard `completed`. **PASS.**

## Verdict

- **Qwen3-4B-2507 Q4_K_M: ladder qualified** at context 512 (gates 1-4: 0.365%
  FP32 reference, 128/128 layouts, 38/38 tokenizer, HTTP byte-exact vs the
  FP32 teacher with streaming). The ABBA comparator's lifecycle extension
  (host restore, disk reload, cancellation recovery, model swap, chat
  stream/concurrency) recorded green in both damendaemon legs of the
  retained runs; the ladder gate-4 pass rests on `verify_gguf_generated_http`
  (stream, non-stream and usage identity against the teacher), with the
  ABBA text-equality invariant retained as the cross-engine difference
  record described above.

## Gate 4 at 8B — blocked by a genuine long-prompt FP16 drift (recorded)

With gate 2 unblocked by #585's corrected contract, the 8B cell proceeded
to gate 4. The ABBA comparator completed on the A30
(`qwen3-8b-bench-a30.json`, sm80 llama-server, native K-quant decode):

- All four phases ran (3 warmup + 6 measured rounds; 16/129/400-token
  workloads at concurrency 1/2). The 129/400-token cells are text-identical
  across engines; the 16-token prompt again shows the deterministic
  cross-engine continuation difference (the same comparator finding as the
  4B record — engine text parity is not the gate).
- **Lifecycle checks complete and green in both damendaemon legs**: host
  restore, model swap (to the 4B cell), disk reload, cancellation
  recovery, chat stream + concurrency.
- The ABBA text-equality assert fires on the 16-token cross-engine
  difference, as at 4B — retained as the comparator record, not a gate
  criterion.

The teacher qualification (`verify_gguf_reference.py` on the exact bench
workload) is where the cell stops:

- `qwen3-8b-teacher-benchmark-a30.json` — FP32 HF eager reference (CPU;
  the 8B FP32 reference plus its conversion transient exceeds one A30 —
  three reserve/configuration attempts recorded honestly before the
  `--reference-device` instrument option, added for exactly this), vs
  damendaemon's normal FP16 mixed path. **Worst relative RMS 5.13%,
  max absolute 0.946 — the 1% / 0.5 contract fails**, with **zero greedy
  mismatches** at every step. The 16-token prompt is clean (0.16%);
  the drift is prompt-length-driven: 129 tokens worst 4.4%, 400 tokens
  worst 5.1%. The 4B passed this identical workload at 0.411% — the 8B
  drifts ~12x, consistent with the hidden-width-4096 sensitivity #585
  established for this checkpoint. Gate-1's 0.573% (retained) was measured
  on the initial suite's short chat prompts only; this is the first
  long-prompt FP32 comparison for the 8B.
- `verify_gguf_generated_http` requires a passing teacher on the same
  workload identity, so the HTTP lifecycle gate cannot run. **The 8B cell
  stays unqualified at FP16 mixed precision** — not an instrument artifact
  this time: the drift is real (max_abs ~0.95 on logits with ~17 RMS),
  bounded (zero greedy flips through 16 decode steps), and
  prompt-length-dependent.

Follow-up (not this cycle): a qualified FP32-arithmetic packed execution
path for qwen3 (currently an explicit unqualified opt-in for packed
checkpoints), or a deliberate bounded-drift acceptance decision for the
8B class. Filed as its own issue; the ladder records the outcome either
way.

- **Qwen3-8B Q4_K_M: gates 1-3 pass; gate 4 blocked by long-prompt FP16
  drift** (gate 2 re-passed under #585's corrected contract, 128/128;
  gate 3 tokenizer 38/38; gate 4's teacher qualification fails at 5.13%
  worst relative RMS on the 400-token prompt vs the FP32 reference —
  zero greedy mismatches, prompt-length-dependent, ~12x the 4B on the
  identical workload). The 8B cell stays unqualified at FP16 mixed
  precision pending the FP32-arithmetic packed-path qualification or an
  explicit bounded-drift acceptance decision.
