# Benchmarking and evidence

Run qualification and benchmark commands without Python optimization (`-O`,
`-OO`, or `PYTHONOPTIMIZE`). Root `verify_*`, `benchmark_*`, and `compare_*` scripts with assertion-based
acceptance checks reject optimized execution before imports or setup, so disabled checks cannot produce
a passing qualification report.

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Compare identical revisions, precision, context, output lengths, and hardware. CPU tests establish behavior, not GPU performance.

The [fixed performance scorecard](https://github.com/filthyrake/dllm/issues/54)
defines the next required workloads and competitive thresholds. These are
targets, not results. Functional, numerical, lifecycle and performance gates
are recorded separately; a missing or failed comparison never counts as parity.

For a warm result, initialize every execution shape the measured workload can
reach, not just one request at each concurrency. Arrival timing can make the
first round process prompts singly and expose a packed shape only during timed
rounds. Record actual group sizes and warmup cost; report first-use shape latency
separately. The [Q4 network comparison](evidence/quantization/cuda/q4-arithmetic/README.md#all-reachable-prompt-shapes-warm)
shows why this distinction matters. Generic `dllm bench` does not do that warmup
for you.

## Historical baseline results

Profiling identified excessive small-kernel launches in the first executor.
Fused normalization, RoPE/cache writes, activation, and attention reduced decode
from 1,364 to 274 kernels per token. Continuous batching now shares weight reads
across requests. Performance is measured on the A2 against the same immutable
FP16 model snapshots, with independent-reference correctness checks. The final
0.5B HTTP run delivered 139 tokens/s after first output (vLLM: 149), and four
concurrent requests delivered 433 aggregate output tokens/s including prefill
(vLLM: 497). These are short-workload measurements, not a broad parity claim.

See the [hybrid and Apple Silicon proof](hybrid-and-mps.md),
[new-model and longer-context results](model-matrix.md), and
[fused and batched runtime evidence](batching-proof.md) for current
HTTP comparisons, correctness gates, model switching, and limitations. The
[first runtime evidence](runtime-proof.md) preserves the original baseline.

## Measure existing servers

Python 3.11 or newer; no third-party dependencies required:

```sh
python3 -m dllm bench --base-url http://localhost:8000/v1 \
  --models model-a --requests 16 --concurrency 4 --max-tokens 128 \
  --label baseline > baseline.json

# Sequential A -> B -> A requests measure user-visible model transitions.
# The target server must already support routing/loading these models.
python3 -m dllm bench --base-url http://localhost:8080/v1 \
  --models model-a model-b model-a --requests 3 --concurrency 1 \
  --label switching > switching.json

python3 -m unittest discover -s tests -v
```

Comparator context preflight bounds the `/props` response to 1 MiB before JSON
parsing and requires an exact integer match for the expected context.

Set `DLLM_API_KEY` if the endpoint needs authentication. The key is not saved in
results. Use `--prompt-file` for a repeatable workload and `--label` to identify
hardware, model revision, quantization, and server configuration. Prompts are
represented by a hash in the report, not saved as text.

Reports include each request, failures, wall-clock throughput, first visible
output latency, and output-token counts when the server reports them. Streaming
chunks are **not** counted as tokens. Missing usage means token throughput is
unknown. First visible output can lag actual first-token computation, especially
with reasoning models or server buffering. Full details: `python3 -m dllm --help`.

By default this is a closed-loop load test: at most `--concurrency` requests are
in flight. `--arrival-rate` instead offers fixed scheduled arrivals, reporting
client-capacity drops without queuing them. Optional client SLO profiles include
fixed-arrival dispatch delay in latency and request-rate budgets; ordinary raw
latencies remain request-relative. See [timing and profile semantics](performance.md)
before comparing reports. Neither mode measures GPU utilization or internal
weight-transfer timing. Switching timings include everything the server does
before responding; resident versus RAM versus disk state must be controlled and
recorded externally. No warmup or eviction is performed automatically.


## CUDA hardware probe

This standalone probe measures pinned-host-to-device transfers and FP16 matrix
multiplication, validates results, and releases its resources on exit. It refuses
to run without exactly one GPU UUID selected. It is not a model runtime or an
LLM tokens-per-second benchmark. Peak allocations are under 512 MiB GPU memory
and approximately 256 MiB pinned host memory, excluding CUDA/library overhead.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j2
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid ./build/dllm-cuda-probe
```

Alternatively, with an installed CUDA toolkit:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 native/cuda_probe.cu -lcublas -o /tmp/dllm-cuda-probe
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid /tmp/dllm-cuda-probe
```

The first A2 measurements and their limitations are in
[the hardware evidence](a2-hardware.md).

## Modern Qwen3.5 engineering comparison

Active performance work targets Qwen3 or newer, prioritizing Qwen3.5.
`scripts/run_modern_abba.py` owns four sequential fresh-server phases on the
authorized single-APU host: dllm, llama.cpp, llama.cpp, dllm. The workload is
fixed at context 4096, input lengths 128/512/2048, 128 output tokens, concurrency
one/four, three warmups and ten measured rounds per phase and cell. Groups of
two/three receive additional warmups. Explicit raw token IDs prevent different
chat templates from silently changing the input. Early EOS or incomplete
streams fail the run; identical EOS logit suppression is confined to benchmark
requests. Logs and partial failures remain in the output directory.

The runner requires immutable model/engine pins and a hardware label. It creates
a new output directory, takes a task benchmark lock, and stops only its own
server process groups on completion or error. This does not reserve the GPU
against other services. No downloads, conversion or compilation should overlap
the timed phases. See [AMD setup and comparator artifact proof](amd.md).

After all four phases complete:

```sh
PYTHONPATH=. python scripts/summarize_modern_abba.py /path/to/abba-results \
  --output /path/to/abba-results/comparison.json
```

The summarizer rejects missing rounds, changed input IDs and changed manifests.
It records per-cell latency and aggregate throughput, within-engine text
stability, and cross-engine exact-text differences. It deliberately leaves
`competitive_pass` unknown: current engine arithmetic differs, client-visible
post-first-output throughput is a proxy, and internal decode/peak-memory
telemetry remains a separate scorecard gate. Server-ready time also has different
semantics: dllm loads on its first request while llama.cpp preloads. It must not
be presented as a matched cold-model-start result.

## Fixed-workload request integrity

The modern HTTP benchmark and long-context runner share the fixed-workload
request collector. Every non-null usage record must contain exact integer
prompt and completion counts matching the requested workload. Conflicting totals,
missing counts, booleans and floating-point substitutes fail the request.
Earlier output and valid usage remain in failed reports; no throughput proxy is
produced for a failure. Null intermediate usage and identical repeats are allowed.

A broken start barrier produces a failed request record, marks
`start_barrier_passed: false`, and releases the deadline waiter without sending
HTTP. Successful participants start measurement after crossing the barrier.
The waiter's signal can indicate failure completion as well as first output;
only the retained output fields establish that output was observed.

ABBA summaries independently validate stored measured requests rather than
trusting phase success alone. Each round must contain exactly its declared
concurrency, exact integer prompt/output usage, and finite positive timing with
first output no later than request completion, and request duration no longer
than the enclosing round. Throughput uses the common declared output length,
which is included in the summary. Historical reports missing required usage
cannot establish a measured comparison; regenerate evidence rather than filling
missing values by assumption. These checks still produce no competitive verdict.

Modern HTTP phase reports, ABBA summaries and long-context checkpoints use a
shared atomic JSON writer. A failed write preserves the previous complete file;
unique temporary files permit a nested signal checkpoint without reusing the
interrupted writer's file. This is atomic replacement, not a guarantee against
power loss. Report validation and hardware qualification remain separate gates.

The fixed-workload collector additionally limits retained output to 16 MiB of
UTF-8 text and 100,000 nonempty fragments per request. It records these capture
limits and retained byte count. Exceeding either bound fails before retaining
the offending fragment; earlier output stays available as failed evidence, the
response is closed and the deadline waiter is released. This is a capture bound,
not a token estimate or total stream deadline; SSE fragments may coalesce tokens.

Long-context request deadlines are irreversible qualification failures. Timer
callbacks are joined before the request verdict is accepted; a response that
finishes during shutdown remains failed if a deadline already fired. Its partial
or complete response evidence and the first expiry reason are retained, without
a successful throughput proxy. Timely responses retain their original verdict.

A long-context request deadline now sends SIGKILL to the runner's owned server
process group, so a stuck graceful shutdown cannot keep the request reader
waiting. This deliberately skips application cleanup after a hard deadline;
normal successful-run cleanup retains its graceful path. Reports retain the
expiry reason and failed evidence. The runner must own the server session; this
mechanism is not for pointing at or stopping an existing shared service.

Long-context fixture preparation is bounded before any server launch: requested
context is limited to 32,768–1,048,576 tokens, and the generated ASCII corpus to
16 MiB per encoding attempt. Corpus growth stops on non-progressing tokenization
or budget exhaustion; selected token IDs must be nonnegative integers. Reports
record the preparation limits and corpus size. The short prompt remains 128
tokens, and the long prompt reserves 512 output tokens. These are client fixture
bounds, not a claim that a checkpoint or device supports those context lengths.

Long-context qualification gates use explicit runtime checks, including overflow
rejection, request success, capacity budgets and release status. They remain
active with `python -O`; a failed warmup is retained and stops before model
management or a passing capacity report.

Explicit SIGINT/SIGTERM interruption also force-stops the runner-owned server
before writing its checkpoint. This prevents checkpoint I/O or graceful server
shutdown from delaying workload termination. Interrupted reports retain partial
evidence and signal identity with `passed: false`; normal successful cleanup is
unchanged. A checkpoint failure cannot skip the stop operation.

Long-context control responses (status, unload and expected overflow errors) are
limited to 64 KiB and must be JSON objects with finite numeric values. Oversized
or malformed responses fail qualification. Overflow error responses are closed
on success and every failure path; the exact expected rejection remains required.

The fixed-workload benchmark validates tokenizer output before contacting HTTP:
all four prompts at each 128/512/2048 length must be available at their original
17-token offsets, and selected IDs must be nonnegative integers. This check is
active under optimized Python; short or malformed corpora do not launch a
measurement or produce a mislabeled workload report.

Runtime environment metadata uses an explicit allowlist of execution modes and
budgets. Credentials, private paths, and unrecognized environment variables are
omitted; report filtering does not change the environment used by subprocesses.
