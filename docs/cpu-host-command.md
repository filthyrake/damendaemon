# Explicit two-process CPU host reference

The command composes TLS endpoints, selected checkpoint-range loading, host-worker
setup and bounded raw completion streams. It is a CPU reference path for text-only
Qwen3.5 snapshots, not a production server or hardware qualification.

Both processes need local copies of the same externally verified snapshot and
compatible tokenizer/generation metadata. Supply its 64-digit hexadecimal identity
through `--model-id` on both sides. This value is an operator assertion; the command
does not hash every weight payload to independently verify it. Setup additionally
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
HTTP admission, persistent workers, remote provisioning, practical model quality,
hardware and network performance remain open under #51.
