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

## Native image and tool policy

`HostFlashImageGenerator` adds separate native still-image admission. Construct
its topology with `native_image_session_identity(text, verified=...)`, explicit
`flash_embeddings` input and the ordinary Flash output-media exclusions. Its
`FlashImageToolText` tokenizer is detached and its atomic image markers, native
tool/text policy, fixed processor contract and verification mode are bound to
that session identity. Ordinary native text owners still reject this mode.

The generator borrows immutable CPU `FlashVision` parameters whose geometry
matches the language snapshot. For a byte-verified language session, use
`load_host_image_vision(snapshot, worker, ...)`; it hashes the bounded complete
snapshot and checks for changes after vision loading. A separately prepared
vision object cannot claim that verification merely because its shape matches.
The processor policy itself is fixed by the session; this is not a claim that
arbitrary processor metadata or arbitrary external image preprocessing was
verified.

`generate_images(messages, images, ...)` and
`generate_image_tools(messages, images, tools, ...)` accept explicit image
markers plus CPU uint8 RGB tensors in prompt order. They validate markers,
tools, processor bounds, context and prepared-language storage before host
execution. Pixels are detached; the coordinator projects vision and replaces
image placeholders once, then uses owned feature completion and the existing
strict native tool parser. No image payload travels to language peers.
Successful responses release slots for reuse; malformed preparation leaves the
host epoch usable. Cancellation or parser failure after execution starts
retires it, and failed cleanup remains retryable. `close()` drops the generator's
vision reference after host cleanup succeeds; callers still own any borrowed
vision references. Ordinary native text/tool methods also remain available on
this explicit owner. Native command and HTTP admission are separate gates.

## Standalone command

Use `scripts.run_glm_flash_host_topology` with `--native-chat --flash-images` on
**every participant**. The complete role accepts one `--messages` JSON file
containing bounded inline PNG/JPEG `image_url` parts, plus optional `--tools`.
Existing plan, stage index, explicit peer addresses, TLS certificates and source
identity arguments are unchanged. Image/text policy disagreement fails the
handshake. Image mode currently rejects raw prompts, batch messages and serve.

Coordinator-only image controls are `--vision-weight-mib` (default 256),
`--image-workspace-mib` and `--image-output-mib` (default 64 each), and
`--max-patches` (default 4096). Ordinary workspace/output controls also bound
vision execution. Worker processes reject image-budget overrides and load only
their assigned language layers. All participants require matching image marker
metadata and the pinned processor configuration.

The coordinator validates the inline request and loads vision before connecting
peers. Optional snapshot verification checks the already-hashed source again
after vision and language loading. Ready metadata reports `still_image`;
completion success is emitted only after every peer acknowledges STOP. No
provisioning, discovery, remote image fetch or hardware execution is implied.
