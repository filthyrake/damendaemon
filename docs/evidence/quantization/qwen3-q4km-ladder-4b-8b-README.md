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
  FP32 teacher with streaming). Note: the ABBA comparator's lifecycle
  extension (host restore, disk reload, cancellation recovery, model swap)
  did **not** execute in the retained runs — the cross-engine text-equality
  assert fires before the lifecycle block writes its checks. The ladder
  gate-4 pass rests on `verify_gguf_generated_http` (which exercises stream,
  non-stream and usage identity against the teacher), not on the ABBA
  lifecycle; a lifecycle-complete comparator pass is open follow-up once the
  benchmark invariant is reconciled with cross-engine text differences.
- **Qwen3-8B Q4_K_M: gates 1 and 3 pass; gate 2 fails on chunked prefill**
  (defect #585, localized: multi-chunk 128-boundary continuation,
  qwen3-8B-specific; single-chunk exact; llama31-8B at the same width passes).
  Gate 4 not attempted at 8B — the layout gate precedes it in the ladder; the
  8B cell stays short of qualification until #585 is fixed and gate 2
  re-passes at the default contract.
