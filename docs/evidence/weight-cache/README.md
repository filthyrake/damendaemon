# Prepared dense weight cache: A2 proof

Pinned sources are recorded in each JSON report. These are fresh Python
processes on the same NVIDIA A2, using PyTorch 2.6.0+cu124, context 2048 and
batch capacity four. The filesystem page cache is warm; this is not a reboot
or cold-disk benchmark. Conversion and cache persistence are included in the
first run. No native C++ Q4 override was enabled for these measurements.

| Mistral 7B Q4 phase | Cache creation | Cache reuse |
|---|---:|---:|
| Host preparation / mmap | 52.213 s | 0.170 s |
| GPU executor construction / transfer | 3.821 s | 4.251 s |
| Load + 1024-token prefill + eight decode steps | 58.417 s | 6.727 s |

The end-to-end measured workload improves 8.68x. Host mmap time alone does not
represent model readiness. Both runs produced identical eight token IDs and
identical live/peak GPU allocation figures. These are individual observations,
not a multi-run median or an inference-throughput comparison.

Qwen2.5-0.5B Q4 and Q8 pass all independent reference checks in both cache
creation and fresh-process reuse; the complete check records are identical
between each pair. Cached Mistral Q4 also passes its independent 4352-context
sliding-window trace. Cache format/type/shape validation, source invalidation,
corrupt-file recovery and atomic publish failure are covered by unit tests.

Cache identity relies on trusted local source file metadata. Tensor payloads
are not cryptographically authenticated. The feature is opt-in, has no automatic
on-disk eviction, and currently applies only to prepared dense Q4/Q8 weights.
