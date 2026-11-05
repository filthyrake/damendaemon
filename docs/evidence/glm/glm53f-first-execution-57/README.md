# #57 — First execution evidence: teacher-forced forward through real GLM-5.3-Flash weights, all gates pass

**Second #57 physical-qualification increment** (after #709): the first cell
to EXECUTE the flash reference forward against the official
`zai-org/GLM-5.3-Flash` weights — 32 teacher-forced tokens through the
family's TWO attention boundary kinds: layer 0 (gated-deltanet linear
attention) and layer 3 (sparse MLA + lightning indexer), via the production
`FlashStageSlots` serving-shaped path.

## Gates (pre-declared) — ALL PASS

1. **Execution**: 32-token prefill completes through both stages; outputs
   finite; shapes exact `(1, 32, 4, 4096)` (4 hyper-connection streams ×
   4096 dim). Layer 0: 245.62 s; layer 3: 372.02 s (fp32 CPU reference,
   8 threads; documented reference cost, not a perf claim).
2. **State parity on real weights**: `prefill(31) + decode(1)` vs full
   `prefill(32)` last-position hidden —
   - layer 0 (linear/gdn): max |Δ| 1.02e-8 (6.42e-6 relative, |h| 1.6e-3)
     — the RECURRENT-STATE decode path reproduces prefill exactly through
     the gated-deltanet state machinery.
   - layer 3 (sparse MLA + indexer): max |Δ| 3.58e-6 (1.18e-5 relative)
     — the sparse top-k indexer + KV decode path reproduces prefill.
3. **Reload determinism**: full stage reload from the NAS reproduces both
   stages' outputs **byte-identically**.

## Real-geometry finding (no code change; disclosed for the next rung)

The KDA (gated-deltanet) layer's workspace demand at real geometry is
~17 MB/token (64 heads × 128 dim): a 32-token prefill needs ~832 MiB
against the loader's 64 MiB default. The loader's
`max_workspace_bytes` is caller-passable with a 2 GiB cap — this cell
passed 2 GiB and fit. **Consequence for practical context**: the 2 GiB
cap admits KDA chunks of only ~115 tokens; a 32K-context chunk-2048
prefill needs ~36 GiB of KDA workspace — the cap is the next
fixture-scale bound to dissolve when the practical-context rung comes.
(Same discipline as #703/#707: dissolve structurally, defaults
unchanged, pin with tests.)

## Honest scope

- Two layers of a 45-layer model, teacher-forced, hidden-stream parity
  only — no logits (mid-model), no generation, no quality verdict, no
  throughput claim. Synthetic deterministic token IDs in real vocab space.
- Vision tower and MTP layer remain excluded/unqualified.
- CPU-only on labnode3 (A30/A2 lanes untouched); NAS read-only; peak RSS
  20.5 GiB; cleanup releases.

## Artifacts

- `glm-cell-c.log` — full gate output
- `glm57_exec_cell_c.py` — the cell (gates pre-declared in the docstring)
