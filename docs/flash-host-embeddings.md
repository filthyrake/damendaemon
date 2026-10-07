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
