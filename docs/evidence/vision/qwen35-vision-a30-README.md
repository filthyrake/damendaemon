# Qwen3.5 vision on the NVIDIA A30 (#579)

Delivers #50's hardware-extension cell for the A30 with real memory/lifecycle
proof. The labnode3 A30 (24,576 MiB, sm80, torch 2.13.0+cu130, idle) ran the
full existing qualification ladder under the guarded shared-GPU reserve gate,
TF32 disabled, with every threshold unchanged:

| gate | 0.8B | 2B |
|---|---|---|
| Encoder intermediate gates | 5/5 cases (`qwen35-vision-a30-enc08b.json`, depth-12 gate 1e-4 relative) | 135/135 checks, worst RMS 1.696e-04 vs 7e-04 gate (`qwen35-vision-a30-enc2b.json`, depth-24) |
| Full-model image gate | 27/27 (`qwen35-vision-a30-model08b.json`) | 54/54, worst RMS 4.614e-03 vs 0.01, greedy all, changed-image control healthy (`qwen35-vision-a30-model2b.json`) |
| HTTP service gate | 20/20 (`qwen35-vision-a30-svc08b.json`, six-case reference) | 20/20 (`qwen35-vision-a30-svc2b.json`) |

Both service runs report persistent retained allocations of 17,039,360 bytes
identified as cuBLAS workspace through the trusted #575 allocator reference;
no unexplained allocation growth. Host budget scaled by the checkpoint
estimate (10 GiB for 2B, 8 GiB for 0.8B).

## Policy change (on merit)

`damendaemon/vision_policy.validate_execution` admits `NVIDIA A30` by name; the
error text names all qualified devices. Thresholds, geometry admission (both
dense Qwen3.5 geometries), context <=1024, <=4 slots, and the
`DAMENDAEMON_EXPERIMENTAL_VISION` opt-in are unchanged. The service gates were
executed with exactly this committed policy. Tests: `tests/test_vision_policy`,
`tests/test_vision_geometry_admission`, `tests/test_vision_load_preflight` —
11 passed, 1 skipped (torch-dependent local skip), run against the modified
policy.

## Runtime note recorded

The first 0.8B service attempt used `qwen35-images-golden.pt` (3 cases) and
failed with an IndexError: the service harness indexes payloads up to
`payloads[5]`, requiring the six-case reference
(`qwen35-images-q8-golden.pt`, sha256 `2d0bf330…`, the same reference as the
historical 0.8B A2 CUDA service evidence). No state was corrupted; retained
for the next person.

Reproduction: guard_shared_cuda on `GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a`,
`scripts/verify_qwen35_vision.py`, `verify_qwen35_images_model.py`,
`verify_vision_service.py` with `DAMENDAEMON_EXPERIMENTAL_VISION=1`.
