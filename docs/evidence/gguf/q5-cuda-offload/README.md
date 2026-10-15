# Q5 CUDA FP16 reference with CPU-offloaded weights

Runner implementation: `05bdc5e`. Fixed historical checkpoint SHA, prompt48,
context64, eight teacher-forced positions including post-EOS. Independent
`gguf.quants` expansion rounded FP16, HF eager Llama, untied embeddings,
TF32 disabled. Accelerate 1.10.0 installed only under `/tmp/dllm-224/oracle-deps`;
`cpu_offload` streams parameters to CUDA and back. All watched linear,
embedding and decoder layer outputs were CUDA at all eight steps (manifest).
The full logit payload remains on labnode3 under `/tmp/dllm-224/oracle/`, with
its SHA-256 retained here; no claim of permanent storage for that scratch file.

The oracle completed in 81.518 seconds under the 120-second shared guard,
peaking at 1,056,964,608 reserved bytes. Incumbent health remained good and A2
free memory returned from 4,255 MiB to 4,255 MiB. Each candidate ran separately
with fresh guard admission and a 1,536 MiB allocator cap, no graphs.

Both explicit mixed and FP32 candidates pass the unchanged <1% relative RMS,
<0.5 absolute and exact greedy gates against this CUDA FP16 oracle. Per-step
errors and guard telemetry are retained. This covers one short fixed prompt,
not the complete original multi-prompt/context/graph suite. The historical
1.0343% native / 1.0454% Triton post-EOS failures were not reproduced here;
weight residency, allocation and the current runtime differ from the historical
run. This is not a demonstrated root-cause fix or closure of parent #45.
No defaults, thresholds or native Q5 dispatch eligibility changed.

Seventeen relevant CPU guard/arithmetic tests passed before the experiment.
No GPU performance or end-to-end throughput claim is made.
