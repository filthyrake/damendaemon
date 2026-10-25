# #624 verification cell: APU admission + first completed 32K request (2026-10-24)

## Verdict

**The APU-aware admission works and the first full 32K request on HOMER
completed: 32,011 input tokens prefill + 512 output tokens generated,
guard completed (max 81.0 °C, watchdog silent, cleanup complete).** The
#68 C1 time bounds were NOT met as specified (first output 1071.8 s vs
≤600 s; completion 1238.2 s vs ≤1200 s) — recorded honestly below; the
completion/output/release halves of C1 now have their first evidence.

## Cell

HOMER gfx1151, guard_homer 93 °C (max observed 81.0 — the entire 32K
chunked-127 prefill + 512-token decode stayed under the 85 limit at 45 W
caps with runtime pacing soft-stop 55), fans fixed@5 verified, host
restored to balanced/120-140-120 W, 29-31 °C idle. Tree
`issue-624-apu-admission` (`246c9d86` + hybrid_precision context fix);
in-process driver (no HTTP) consuming the full event stream with
`ignore_eos=True` for the bounded diagnostic.

## Admission record (`admission.json` via DAMENDAEMON_ADMISSION_REPORT)

Required 52.74 GiB admitted under the 6 GiB APU reserve where the
discrete 16 GiB reserve refused the identical request (45.24 GiB bound,
the #68 C1 blocker). Allocator cap 56 GiB; no other term changed.

## C1 numbers (single client, temperature 0, ignore_eos for the diagnostic)

| metric | value | #68 C1 bound |
|---|---|---|
| input tokens | 32,011 | 32,256 target (paragraph-aligned) |
| first visible output | 1071.8 s | ≤600 s — **not met** |
| completion | 1238.2 s | ≤1200 s — **not met** |
| output tokens | 512/512 | complete |
| peak guard temp | 81.0 °C | under 93 cutoff |

First events show special-format tokens (248046, 198, 248045...) with
empty text before prose begins — the earlier "silent stop" at one event
was the model's immediate EOS-family format tokens, not a runtime fault;
`ignore_eos` was used only for this diagnostic and the output text is the
model meta-describing the repeated-paragraph prompt (no chat template on
a raw completion), which is legitimate raw-completion behavior.

## Honest boundaries

- Prefill throughput (~30 tokens/s at chunk 127) is the remaining gap, not
  stability, memory or thermal collapse; the time bounds need either the
  grouped-attention/prefill-throughput work to land in the serving path or
  larger chunk sizes, which is #68's own qualification question.
- Single client, batch 1, no concurrency claim; production admission stays
  512 (this run used the explicit experimental context opt-in).
- The watch terms: host reserve held (watchdog silent, 10 Hz), MemAvailable
  never below the reserve during the run.
