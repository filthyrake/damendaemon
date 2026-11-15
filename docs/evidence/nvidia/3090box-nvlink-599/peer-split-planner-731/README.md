# #731: interconnect-aware peer-resident split placement — planner validation

2026-11-14, 3090box (2× RTX 3090, NV4), guarded (`guard_shared_cuda`, both
UUIDs). Policy module `damendaemon/peer_split_policy.py` (pure, injected
tests `tests/test_peer_split_policy.py`), validation runner here.

## Result: ALL GATES PASS (attempt 2)

| Gate | Verdict |
|---|---|
| live_inventory | PASS — 2× RTX 3090, mutual peer access both directions |
| plan_picks_split | **PASS** — [58,66) selects `peer_resident_split` on pair (0,1) at the measured floor |
| arithmetic_matches_730 | PASS — projected peak == 24,891,097,088 B exactly (#730's measured peak) |
| nine_layers_refused | PASS — 9th layer (disclosed layer-58-size proxy; layer-66's bank is on homer) refuses on headroom, arithmetic retained |
| opt_in_off_by_default | PASS — `DAMENDAEMON_PEER_RESIDENT_SPLIT` unset → `host_resident_stream` |

## Attempt 1 (retained, planner-1.json): the refusal that caught the denominator error

`plan_picks_split` FAILED while `arithmetic_matches_730` PASSED — the
honest tell. The policy's 3% assumed free-fraction floor exceeded the
TRUE headroom because the RTX 3090 exposes `total_memory` =
**25,351,356,416 B (~23.6 GiB)**, not the 24 GiB nominal the earlier
figures divided by. Corrected #730 headroom: **1.82% / 2.58%** free at
peaks 98.18%/97.42% of real device memory (corrected on #730 and in the
banksplit-730 evidence README).

The policy now derives its floor from the measurement itself:
`DEFAULT_MIN_FREE_FRACTION = 1 − 24,891,097,088 / 25,351,356,416 ≈ 0.0182`
— a projection is admitted only when it leaves at least what the PROVEN
#730 cell measurably demonstrated. Evidence-derived, not assumed.

## What shipped

- `damendaemon/peer_split_policy.py` — pure decision module: strict
  tri-state opt-in (`DAMENDAEMON_PEER_RESIDENT_SPLIT`, default off until
  a promotion issue), mutual-only P2P pair extraction ordered by
  aggregate memory, contiguous-half split projection (per-GPU banks +
  measured working set → projected peak → free fraction per device),
  every refusal carries its arithmetic. No torch, no sysfs — callers
  pass the `topology_inventory` dict and byte numbers.
- 23 unit tests, all injected (tri-state, one-directional-pair refusal,
  aggregate ordering, measured-range select, 9-layer headroom refusal,
  small-stage no-split, no-pair stream, empty inventory).
- Integration point for consumers: the #604 inventory feeds
  `peer_split_decision(layer_bytes, inventory)`; the #719 fleet chain and
  future stage planners consume the recorded arithmetic.

No throughput/SLO claim — placement-policy validation only. Promotion of
the opt-in to a default requires more cells (per the issue's gate 4).

Parents: #731 (this cell), #730 (the measured envelope it consumes),
#599 (interconnect-aware placement).
