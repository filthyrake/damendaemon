# Completion streams over prepared CPU host workers

`dllm.host_completion.HostCompletion` exclusively owns an already authenticated,
handshaken rank-0 `HostHybridWorker`. The peer owner must keep calling rank 1's
`step()` until its session closes. The adapter creates no listener or peer thread.

```python
from dllm.host_completion import HostCompletion

runner = HostCompletion(coordinator, eos_ids=text.eos_ids)
try:
    for event in runner.generate_batch(
        [[1, 2, 3], [4]], max_new_tokens=[8, 4], chunk_size=2
    ):
        print(event)
finally:
    runner.close()
```

The examples use token IDs; callers must encode them with the tokenizer bound to
the workers' mapped vocabulary policy. Both peers must already agree on their
model identity, precision, layer split, context, slot and token policy contracts.
Do not call worker commands directly after transferring ownership to the adapter.

Requests contain 1–4 ragged prompt rows, further bounded by negotiated session
slots. Output limits are 1–4096 per row; prompt plus output must fit the context.
Chunk size is 1–4096 and activation chunks/decodes must fit the negotiated frame
cap. Invalid input fails before dispatch and leaves an idle session usable.
Sampling is the worker's greedy policy; EOS IDs must be mapped and not excluded.
The adapter neither implements chat templates nor performs model tool calls.

Events contain `row` and `token_id`. `generate()` wraps one row and omits `row`.
Completed rows are released on both peers before final success events, retaining
original slot identity while survivors decode. Usage includes the sampled EOS.
Successful requests leave empty slots available for the next request; weights and
transport remain resident. Only one request may run at a time. This is sequential
session reuse, not continuous admission or concurrent HTTP serving.

Close abandoned iterators. Cancellation callbacks run between commands; blocked
I/O retains the transport's existing timeout. Calling `runner.close()` from
another thread interrupts blocked I/O immediately. Errors, cancellation and early
iterator close discard the entire uncertain session rather than reusing partial
state. The original error is preserved if cleanup fails; retain the runner and
retry `close()` until stage cleanup succeeds. No final success is emitted before
all releases succeed, although token events already returned are provisional.

Authenticated local peers match whole CPU pipeline token sequences for floating,
Q4_0 and Q8_0 checkpoints, including ragged chunks, unequal limits and reuse.
These tests do not qualify official-model quality, practical long context,
hardware, network throughput or multi-user SLOs. Parent issue: #51.
