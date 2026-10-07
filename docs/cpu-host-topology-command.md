# Explicit multi-process CPU topology

`python -m scripts.run_cpu_host_topology` runs one CPU stage per process using
existing bounded checkpoint loading, mutual TLS and the explicit topology owner.
It supports text-only Qwen3.5 reference snapshots with `none`, `q4_0` or `q8_0`
weights and CPU FP32 computation. It does not discover hosts, provision services,
copy models, issue credentials or select accelerators.

Create a local JSON plan containing exactly `ranges`, with 2–8 nonempty,
contiguous ranges covering every layer. For a four-layer fixture, `plan.json` is:

```json
{"ranges": [[0, 1], [1, 3], [3, 4]]}
```

Use the actual layer count and suitable ownership ranges for your checkpoint.
Every process needs the same plan, model ID, tokenizer/EOS metadata, precision,
context/limit, slots, prefill chunk, frame cap and epoch. The optional trusted
`--chat-template` content and `--verify-model-id` mode also bind the session
identity. Mismatches fail setup; there is no partial-topology admission or retry.

Start one worker for every nonzero index. For index 1:

```sh
python -m scripts.run_cpu_host_topology worker /path/to/snapshot \
  --plan /path/to/plan.json --index 1 --address 127.0.0.1 --port 9501 \
  --model-id "$SNAPSHOT_SHA256" --context 512 --slots 2 --prefill-chunk 64 \
  --ca /path/to/ca.pem --cert /path/to/worker1.pem --key /path/to/worker1.key \
  --peer-cert /path/to/coordinator.pem
```

Start index 2 separately with its own port and certificate. Workers accept one
coordinator, close their listener after authenticated acceptance, load only their
assigned tensors and wait for the shared handshake. `listening` is not model
readiness. Port 0 requests an available local port and reports it in that event.

The coordinator's `peers.json` lists every remote stage in ascending index order:

```json
[
  {"stage": 1, "address": "127.0.0.1", "port": 9501,
   "server_name": "localhost", "peer_cert": "/path/to/worker1.pem"},
  {"stage": 2, "address": "127.0.0.1", "port": 9502,
   "server_name": "localhost", "peer_cert": "/path/to/worker2.pem"}
]
```

Addresses must be literal IPs. TLS hostname verification uses `server_name`, and
exact certificate pins use `peer_cert`; relative certificate paths resolve from
the peers file's directory. Plan/peer JSON is limited to 1 MiB and rejects unknown
fields. Each process supplies its own local CA/certificate/key. There is no
plaintext fallback or implicit wildcard binding.

Run a coordinator completion:

```sh
python -m scripts.run_cpu_host_topology complete /path/to/snapshot \
  --plan /path/to/plan.json --peers /path/to/peers.json \
  --model-id "$SNAPSHOT_SHA256" --context 512 --slots 2 --prefill-chunk 64 \
  --ca /path/to/ca.pem --cert /path/to/coordinator.pem --key /path/to/coordinator.key \
  --prompt 'Hello' --max-new-tokens 16
```

`--tokens` and ragged `--batch-inputs` use the existing
[CPU completion formats](cpu-pipeline-completion.md). Token events are provisional;
final text/usage is emitted only after every worker acknowledges STOP. Failed
connections, stage construction, contract checks and execution close all acquired
owners. Interrupted epochs are discarded, never replayed or resumed.

For HTTP, replace `complete` with `serve`, omit request input, and select
`--http-port`. Binding remains loopback. Existing API/admin authentication,
bounded queue, client turns, interleaved prefill, heartbeat, idle timeout and
`--stop-after-drain` options apply. Status reports every stage range. Administrative
drain finishes admitted work and acknowledged slot release before optional STOP.
SIGTERM interrupts all links and closes HTTP; workers see failed epochs rather
than a fabricated STOP acknowledgement.

`--model-id` is an operator assertion by default. With `--verify-model-id` on every
process, each independently hashes its local snapshot before connecting/binding,
checks the model ID and pins the resolved root. After local loading, the command
rechecks the selected file inventory/stamps before handshake. Identity byte/time
limits and regular-file restrictions match the [two-process command](cpu-host-command.md).
The coordinator cannot attest what a remote process loaded; HTTP metadata says
only whether its own snapshot was verified. Plan/peer files outside the snapshot
are preferable; identity ignores unrelated files but bounds root directory size.

Each stage independently enforces weight/cache/loader/projection/output workspace
budgets. `--timeout` is a per-connection/per-frame deadline, at most 30 seconds,
not a whole-request deadline. Start processes within their accept deadlines and
budget startup for snapshot verification/loading. Startup is sequential across
explicit peers; there is no availability retry or automatic restart.

Three-/four-process loopback fixtures cover numerical completion equality,
HTTP/SSE/drain, mismatched source/session contracts, partial setup failure and
signal/worker loss. They establish software composition only. Practical model
quality, long-context capacity, actual network behavior, GPU execution and SLOs
remain unqualified.
