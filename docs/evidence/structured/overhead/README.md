# Isolated grammar setup overhead (#220, parent #15)

CPU-only measurements on labnode3 using the real Qwen2.5-0.5B-Instruct tokenizer,
revision7ae557604adf67be50417f59c2c2f167def9a775, with tokenizer SHA and dependency
versions in `cpu-setup.json`. No model weights, CUDA context or inference were
loaded. Production `Constraint` constructor is timed with one PyTorch/BLAS
thread; XGrammar retains its production two-thread compiler configuration.

Three fresh model/compiler instances and five repeats per instance yield:

| Operation | Median |
| --- | ---: |
| First Constraint, including tokenizer/compiler setup | 2.052402 s |
| Same schema with warm compiler cache | 44.972 microseconds |
| Changed schema, existing compiler | 2.194562 ms |
| Unconstrained option preparation | 2.348 microseconds |

The constructor includes matcher and mask allocation; these are not isolated
compiler-kernel timings. Changed schemas use distinct integer constants. Raw
samples are retained, including first-use variability. The host also ran an
independent bounded workload, so these are observed shared-host CPU costs, not
exclusive latency guarantees. Model activation is excluded from every sample.

This closes the bounded CPU-setup measurement task. It does not measure GPU
mask application, full generation overhead, throughput, or SLOs. Parent #15's
matched runtime-overhead gate remains open.
