# Packed prefill across same-host GPU stages

Tracked in GitHub issue #51; the expanded CUDA Q4 numerical gap also belongs to
#23. The layer-partition executor now shares the existing packed-prefill math
with the single-device batch executor. It combines prompt rows for linear
layers while keeping attention and KV ownership isolated by prompt and slot.
Each group's total rows are bounded by one configured context; larger groups
split without changing order. The same-host FP16/Q8 path enables this by default.
Q4 remains sequential by default. `DLLM_PARTITION_PACKED_PREFILL=0` selects the
sequential control; `1` explicitly enables packing for experiments at any of the
three supported precisions. This does not yet add distributed packed prefill.

## Correctness and lifecycle

Pinned Qwen2.5-0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`,
A2/A30 split 12/12, context 512, four slots:

- FP16, Q8 and Q4 each pass the original 82 independent logit/token checks.
- Expanded short/mixed-prefix traces pass 282 checks each for FP16 and Q8.
  Both stages execute packed prefill on 11 groups (22 observed stage calls).
- Q4 fails three of the expanded 282 checks in **both packed and sequential
  controls**, with identical failing records. Numeric RMS/maximum error remain
  within the original limits, but exact greedy IDs differ at near-tied FP16
  logits. There are no additional greedy mismatches from packing. This is not
  labeled a passed gate, and no tolerance or reference dtype was changed.
- All three precisions pass four alternating weight sessions plus fresh recovery
  after a partial upload failure: 410 independent checks per precision, stable
  graph storage during rebind, and full host/GPU release afterward.
- Real same-host HTTP qualification passes 19 checks per precision with packing
  explicitly enabled, including streaming, concurrency, cancellation and swaps.
- The shared-math extraction also passes existing Apple M4 gates: Qwen2 Q4/Q8
  and Llama Q8 each 282 checks, Qwen3 Q8 284 (1,130 total).
- 434 local tests pass, including rejection before GPU/cache work for invalid
  packed groups and the conservative precision-gated default.

The three Q4 failures occur at trace events 61, 135 and 159, during decode. For
example, the native result ties token IDs 1251 and 3589 at 15.5625, while the
independent reference favors 3589 over 1251 by 0.015625. Both controls show the
same failures. `qwen2-partition-{packed,sequential}-tiny-q4-diagnostic.json`
records full metrics and top logits. The broader gap remains open; this change
does not claim to fix it.

## Matched Q8 HTTP measurement

The same-host server runs sequential/packed/packed/sequential in separate
launches, keeping Q8 weights, A2/A30 placement, context 512, four decode slots,
16 raw input IDs, 32 output tokens, and **zero idle coalescing delay** identical.
Each concurrency gets one warmup and five measured rounds per launch: ten
measured rounds per mode. Startup/capture is excluded; full HTTP delivery is
included. All generated text and usage match between modes.

| Concurrent requests | Prefill | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Sequential | 27.66 | 138.90 | 229.68 |
| 1 | Packed | 25.60 | 131.42 | 242.63 |
| 2 | Sequential | 51.61 | 210.70 | 300.89 |
| 2 | Packed | 38.83 | 202.88 | 310.68 |
| 4 | Sequential | 97.80 | 277.89 | 454.28 |
| 4 | Packed | 59.38 | 239.08 | 530.72 |

At concurrency four, median first text falls **39.3%**, request latency falls
**14.0%**, and aggregate throughput rises **16.8%**. A one-request group uses the
same underlying single-prompt path in both modes; its smaller measured difference
is treated as variability, not a packing speedup. Existing services remain on
the shared GPUs. These results do not compare different engines, precisions,
placements or the distributed service.

The same counterbalanced measurement was also run separately for FP16 and Q4,
changing only the packing flag within each precision. At concurrency four:

| Precision | Prefill | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---|---|---:|---:|---:|
| FP16 | Sequential | 86.40 | 279.27 | 451.81 |
| FP16 | Packed | 54.66 | 239.41 | 525.53 |
| Q4 | Sequential | 97.44 | 267.85 | 470.40 |
| Q4 | Packed | 60.57 | 233.09 | 544.14 |

These are 16.3% and 15.7% throughput improvements respectively. Gains are not
uniform: FP16 concurrency two improves first text from 43.31 to 33.27 ms but
throughput falls from 292.55 to 280.87 tokens/s (4.0%). The raw `*-none.json`
and `*-q4_0.json` benchmark reports retain every concurrency and trial. Q4's
matching HTTP text does not resolve its expanded numerical gate failures;
packing remains experimental and disabled by default for that precision.

Reproduce with `scripts/verify_partition_packed.py` and independent traces from
`verify_stress.py --reference-only --tiny-prefixes`. The replay helper persists
failed checks too and offers `--sequential` for the exact-trace control.
`verify_partition_rebind.py` and `verify_partition_service.py` run with
`DLLM_PARTITION_PACKED_PREFILL=1`. Use
`scripts/experiments/bench_partition_prefill_service.py` for the counterbalanced
HTTP measurement. Explicit benchmark mode selection makes the control independent
of later changes to the default policy. Run GPU jobs sequentially.

A final Q8 HTTP run without the opt-in environment flag also passes all 19
checks (`qwen2-partition-packed-default-http-q8.json`). Final lab allocations
return to A2 0 MiB and the existing A30 service at 13,403 MiB. No network
firewall changes are needed for this same-host feature.
