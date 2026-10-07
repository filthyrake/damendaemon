# Server token observations

Ordinary and batched runtime final `timing.token_observation` reports use
constant storage per request. `scope: server_sample_observation` describes host
observations after a sampled token becomes available and before text formatting.
All sampled tokens count, including EOS, reasoning and tokens with empty text.

`sampled_tokens`, `first_sample_seconds` and `last_sample_seconds` describe these
observations relative to request start. With at least two sequential observations,
`mean_gap_seconds`, `max_gap_seconds` and `interval_tokens_per_second` describe
first-to-last sample intervals. A zero interval has no rate. A single token has
no gaps or interval rate. No percentile is estimated from these summaries.

These intervals include host scheduling, sampling bookkeeping, prior-token text
formatting and backpressure. In ordinary generator execution, consumer pauses
between tokens are included; the consumer's final resume does not extend the
last-sample timestamp. Batched observations also include contention and queue
backpressure. They are neither isolated GPU/kernel decode timings nor client
visible time to first output or inter-output gaps. In particular they cannot
satisfy the scorecard's `engine_token_steps` qualification requirement.

If speculative verification emits a block, `sequential_token_timing` becomes
false for the entire request. All three gap/rate fields become null, even if
later tokens are sequential. Counts and observed first/last offsets remain;
burst emission timestamps cannot recover the hidden per-token execution times.

Legacy `first_token_seconds`, `total_seconds` and `decode_tokens_per_second`
fields retain their behavior. This software instrumentation makes no measured
hardware performance or user SLO claim; those remain tracked under GitHub #59.
