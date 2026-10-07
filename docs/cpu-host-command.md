# Explicit two-process CPU host reference

The command composes TLS endpoints, selected checkpoint-range loading, host-worker
setup and bounded raw completion streams. It is a CPU reference path for text-only
Qwen3.5 snapshots, not a production server or hardware qualification.

Both processes need local copies of the same externally verified snapshot and
compatible tokenizer/generation metadata. Supply its 64-digit hexadecimal identity
through `--model-id` on both sides. This value is an operator assertion by default; `--verify-model-id` independently
hashes the bounded local snapshot before connecting, as described below. Setup additionally
binds tokenizer serialization, EOS, model configuration, quantization, token policy,
context, slots and layer coverage. No checkpoint is downloaded or transmitted.

Use existing certificates from your trusted CA. Both sides require TLS 1.3 mutual
authentication, exact peer certificate pins and dllm ALPN. The coordinator also
checks the worker certificate's hostname. Key files stay local; the command does
not issue credentials, print keys, accept plaintext or install services.

Start the output worker explicitly, for example:

```sh
python -m scripts.run_cpu_host_reference worker /path/to/snapshot \
  --address 127.0.0.1 --port 9501 --split 2 \
  --model-id "$SNAPSHOT_SHA256" \
  --ca /path/to/ca.pem --cert /path/to/worker.pem --key /path/to/worker.key \
  --peer-cert /path/to/coordinator.pem --context 512 --slots 2
```

Then run the coordinator in another terminal:

```sh
python -m scripts.run_cpu_host_reference complete /path/to/snapshot \
  --address 127.0.0.1 --port 9501 --server-name localhost --split 2 \
  --model-id "$SNAPSHOT_SHA256" \
  --ca /path/to/ca.pem --cert /path/to/coordinator.pem --key /path/to/coordinator.key \
  --peer-cert /path/to/worker.pem --context 512 --slots 2 \
  --prompt 'Hello' --max-new-tokens 16 --prefill-chunk 64
```

Use the real split for your checkpoint: the coordinator owns layers `[0, split)`;
the worker owns `[split, layer_count)`. Addresses are explicit literal IPs; there
is no DNS discovery or implicit wildcard binding. Worker port 0 chooses an
available port and reports it in the `listening` event. That event means only that
the listener exists. `ready` follows model loading and successful peer setup.

`complete` accepts exactly one `--prompt`, `--tokens` JSON file or `--batch-inputs`
JSON file, with the same bounded format as the [whole CPU command](cpu-pipeline-completion.md).
Batches may have different prompt lengths and per-row output limits. Completion
JSON lines contain sampled tokens, then final text/usage for each row. Final success
is held until the worker acknowledges STOP; both processes then exit. Prior token
events remain provisional if a peer fails. The worker accepts one connection and
closes its listener after authentication; it is not a persistent service.

Both sides must explicitly select the same `--quantization` (`none`, `q4_0`, or
`q8_0`), `--epoch`, `--context` and `--slots` (1–4). Context above 512 also requires
an explicit `--context-limit`; metadata admission is not long-context qualification.
`--frame-mib` bounds each activation frame, at most 64 MiB. `--timeout` bounds each
connection/transport operation, at most 30 seconds; it is not a whole-request wall
time limit. There is no automatic retry or recovery of failed epochs.

Each process independently enforces `--weight-mib`, `--loader-workspace-mib`,
`--cache-mib`, `--workspace-mib`, `--projection-workspace-mib` and `--output-mib`.
Each option is capped at 2048 MiB. Defaults are small reference budgets, not an
assertion that the full official model fits. Invalid request/endpoint metadata
fails before network or checkpoint payload work. Loader header and memory
admission still occur locally before selected payload reads.

Separate authenticated local processes match whole CPU completions for floating,
Q4_0 and Q8_0 fixtures. Wrong identity/pins and accept timeouts fail without success.
Loopback HTTP admission and explicit resident peer lifetimes are implemented
below. Multi-stage processes use the separate [topology command](cpu-host-topology-command.md).
Remote provisioning, practical model quality, hardware and network performance
remain open under #51.

## Loopback HTTP service

Use `serve` instead of `complete` on the coordinator to keep the authenticated
session resident behind the existing HTTP request owner:

```sh
python -m scripts.run_cpu_host_reference serve /path/to/snapshot \
  --address 127.0.0.1 --port 9501 --server-name localhost --split 2 \
  --model-id "$SNAPSHOT_SHA256" \
  --ca /path/to/ca.pem --cert /path/to/coordinator.pem --key /path/to/coordinator.key \
  --peer-cert /path/to/worker.pem --context 512 --slots 2 \
  --http-port 8088 --alias model --queue-size 8 --queue-timeout 120 \
  --heartbeat-interval 5 --timeout 10
```

Start the same `worker` command first. The serve role takes no prompt/input file.
It reports `http_ready` only after authenticated model setup and binding its HTTP
listener to `127.0.0.1`. `--http-port 0` selects an available port and reports the
URL. TLS secures the worker link; the HTTP listener stays loopback-only and uses
the existing `DLLM_API_KEY`, admin and named-client authentication/admission
configuration. Invalid authentication configuration fails before peer connection.

Submit raw text or token IDs to `/v1/completions`; SSE and per-request usage are
supported. Chat/tools and model changes remain unsupported. See
[CPU host serving](cpu-host-serving.md) for exact request, ownership and metadata
boundaries. This command does not broaden production GPU admission.

`--idle-timeout` optionally stops the owner after 0–86400 idle seconds (0 disables).
`--heartbeat-interval` must be positive, at most 10 seconds and smaller than the
transport `--timeout`; choose enough margin for normal CPU scheduling and I/O.
The worker also has its own operation timeout: it must exceed the coordinator's
heartbeat interval with margin. There is no automatic reconnect or state replay.
Normal idle shutdown sends STOP and closes both peers. SIGTERM or Ctrl-C closes
the HTTP service and aborts the uncertain session; the worker reports EOF/failure
rather than a fabricated graceful STOP. All admitted requests remain bounded by
the shared queue, context, slot and output limits.

Local subprocess tests cover real authenticated HTTP/SSE, authentication failure,
normal idle STOP, SIGTERM cleanup and invalid serving preflight. No service was
deployed to remote hardware, and no GPU or full-model performance is claimed.

### Optional local source identity verification

`python -m scripts.fingerprint_cpu_snapshot SNAPSHOT` prints a versioned SHA-256
identity and per-file digests. Pass that `model_id` to **both** peers with
`--verify-model-id --model-id HEX`. Without the verification flag the identifier
remains an operator assertion. Mixing modes fails session agreement.

The fingerprint binds the selected safetensors shards, their relative paths,
`config.json`, `tokenizer.json`, and any generation/tokenizer configuration and
safetensors index. It includes source bytes, not merely decoded tensor values;
repacking or changing JSON whitespace changes the identifier. Unrelated files
are ignored. No tensor decoding or download occurs. Files must be regular and
contained below the resolved snapshot directory; symlinks below that root,
including Hugging Face cache blob links, are rejected. Use a materialized local
snapshot. The utility does not copy it automatically.

Defaults cap total source bytes at 2 GiB and hashing time at 60 seconds. Explicit
utility `--max-mib`/`--timeout` and peer `--identity-max-mib`/`--identity-timeout`
can increase those caps to 1 TiB/3600 seconds. Reads use at most 1 MiB buffers;
there are also 256-shard, 1024-root-entry and metadata limits. The deadline is
checked between filesystem operations; it cannot interrupt a blocked filesystem
syscall. An oversized snapshot fails before shard hashing, rather than silently
verifying only the stage-local subset.

Each peer hashes before connecting, then checks inventory, inode, size and
nanosecond modification/change timestamps after metadata preparation and stage
loading, before handshake. Source changes abort the owned session. Readiness
reports `model_identity_verified` and `source_model_id`; the transport's
`session_identity` additionally binds effective tokenizer/EOS and verification
mode. This detects ordinary local source changes, not malicious privileged
filesystem manipulation or remote attestation. It does not qualify the model's
quality, context capacity or performance.

HTTP serving commands retain non-daemon request threads through quota release. They wake runtime consumers before joining HTTP threads and report `stopped` only after owned HTTP/runtime cleanup succeeds. Signal interruption still discards peer links rather than claiming an acknowledged worker STOP; failed remote quota cleanup remains charged and prevents success. Incomplete request bodies can hold cleanup until the existing read timeout.
