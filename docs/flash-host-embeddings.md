# Prepared Flash embeddings across CPU hosts

`HostTopology(..., input_kind='flash_embeddings')` explicitly negotiates a
prepared input mode for Flash. Its identity differs from the default token
mode. Other model families reject it. Existing token-mode identities remain
unchanged.

The ready `HostPipelineCoordinator.prefill_embeddings(values, slot=0, start=0)`
accepts CPU FP32 `[sequence, hidden_size]` values. It checks finite values,
chunk/frame/first-stage output limits and the exact slot cursor before peer
admission. It detaches input storage, expands Flash residual streams once,
and forwards ordinary full-stream activations. Remote stages receive neither
pixels nor placeholder tokens. Normal token decode, output media exclusions,
release/reuse and acknowledged STOP still apply.

This is a low-level prepared-input API: it does not establish image provenance,
processor agreement or native chat syntax. Native text/tool owners reject this
mode; native image completion and serving require separate admission. Callers
own preparation and must keep the existing coordinator/peer lifecycle. A
failure after dispatch retires the epoch; retain owners when cleanup needs a
retry. Invalid input detected before dispatch leaves the session usable.

CPU fixture tests compare two/three-stage TLS results with full-model prepared
embeddings and a real vision projection. These are correctness controls, not
remote deployment, official-model quality or accelerator performance evidence.

## Owned completion

Transfer a ready coordinator to `HostCompletion`. In prepared-input mode,
`generate_embeddings(values, max_new_tokens=..., chunk_size=...)` completes one
row; `generate_embedding_batch(rows, max_new_tokens=[...], chunk_size=...)`
accepts up to the negotiated slot count, including unequal prompt/output lengths.
Rows are detached before the first dispatch. Their aggregate FP32 storage must
fit the first stage's output budget, and every expanded chunk and decode batch
must fit its output/frame limits. Full prompt plus requested output must fit
context; no budget is silently reduced.

Events contain token IDs followed by per-row usage and a finish reason. Usage
counts prepared language positions, including image feature positions. Each
finished slot is released before final success. Ordinary token completion
remains available on the same low-level owner. Invalid feature preflight leaves
it usable; failure after dispatch, cancellation during execution or abandoned
iteration retires the epoch. `close()` interrupts blocked transport; retain the
owner and retry `cleanup()` when resource release fails.
