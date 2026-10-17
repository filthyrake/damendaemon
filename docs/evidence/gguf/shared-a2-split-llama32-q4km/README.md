# Real llama-architecture Q4_K_M split-GGUF gate (#498, parent #43)

Source checkpoint: the pinned local llama-3.2-1B import
`/home/damen/dllm-dev/baselines/llama32-1b-import-q4_k_m.gguf`
(llama architecture, Q4_K_M, 807694624 bytes). The retained upstream-writer
preparation created two standard numbered shards
(`split_max_tensors=100` over 147 tensors) and `split_gguf.prepare` verified
exact packed bytes and metadata across all tensors, single versus split, with
raw prompt IDs `[128000, 9906]`, context 128, four output tokens.

## Containment-deadline extension (PR #499)

Two prior attempts (guards retained: `gate-guard.json`, `retry-guard.json`,
plus their `result.json`) completed both numerical traces but hit the
120-second outer deadline sized for the 0.5B Q8 checkpoint — the 1B import
plus four fresh cold processes needs more wall time. PR #499 made the
deadline explicit and bounded: `prepare(timeout_seconds=...)` validated
120..300, carried in the manifest and enforced by `execute`'s remaining-budget
check; the guard's outer cap rose 120→300. The allocator cap (1536 MiB),
memory/temperature/identity gates and all precision gates are unchanged.

## Passing run

`gate300-guard.json` + `passed-result.json` (merged as `result.json` here) must
be read together: guard completed in 144.7 seconds; all four phases
(`single-trace`, `split-trace`, `single-http`, `split-http`) passed at
`timeout_seconds=300`. Peak reserved 830472192 bytes (under the 1536 MiB cap).
Single and split traces match all four normalized-logit SHA256 hashes and the
greedy tokens `[11, 358, 2846, 3411]` exactly. Both HTTP reports pass explicit
load, nonstream output, complete SSE stream, unload and disk reload with
identical text and usage. After the run the guard reported 14965 MiB free and
49 C. No reset, service modification or incumbent interference occurred (the
host is idle; guard ran in optional-incumbent mode).

Scope: one pinned llama-architecture Q4_K_M checkpoint, one bounded trace,
context 128, four tokens — containment checks and lifecycle proof, not a
performance or precision-quality claim. The wider format/architecture matrix
stays open under #43.
