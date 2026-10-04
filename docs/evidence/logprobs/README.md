# Generated-token log probabilities

Issue #28. Chat completions accept `logprobs: true` and optional
`top_logprobs: 0..20`. The default path does not compute scores or copy score
vectors to the host. Requested scores are FP32 log-softmax of original model
logits, before penalties, bias, temperature, top-k/top-p or grammar masking.
This follows the raw-logprobs behavior in the local vLLM sampler reference;
dllm does not import its runtime.

Scores are attached to selected non-EOS tokens, including tokens whose text is
buffered until a UTF-8 sequence completes. ByteLevel and byte-fallback tokenizer
pieces retain their exact bytes; unknown decoder types use nullable bytes.
The per-model presentation cache is bounded and does not retain evicted models.
Streaming and nonstream responses use the chat `logprobs.content` shape.

Initial scope excludes raw-completion logprobs, prompt logprobs and custom text
stops combined with scores. Those combinations are explicitly rejected. EOS
still counts toward completion usage but is not a visible scored content token.
Scores describe the model distribution, not the filtered sampling distribution.

Independent qualification:

- Accurate Qwen3-0.6B Q4 on A2: 24 selected-token checks against CPU FP32 eager
  inference and independent GGUF Q4 conversion; maximum absolute log-probability
  error 0.00009454. All top-five token identities and scores pass.
- Qwen2.5-0.5B on M4: the same numerical and real-service proof with the Apple
  runtime and CPU FP32 reference (exact metrics in the accompanying JSON).
- Both exercise greedy, seeded top-k/top-p sampling and forced logit bias. Bias
  changes selection but leaves reported raw probabilities unchanged.
- Real HTTP streaming/nonstream scores match; four concurrent requests preserve
  scores/output; scoring does not change output; prefix hits, chunked prompts,
  cancellation and recovery pass. A2 additionally uses a 2048-token page pool.
- The local suite has 168 passing tests, including independent log-softmax,
  sampling invariance, request isolation, Unicode byte fragments, protocol
  validation and decoder lifetime.

Initial A2 serving overhead, Qwen2 Q4, paged pool/context 2048, same prompt,
64 outputs and five measured warm rounds:

| Concurrent requests | Scoring off tokens/s | Initial top-five tokens/s |
|---|---:|---:|
| 1 | 205.22 | 179.19 |
| 2 | 351.42 | 300.93 |
| 4 | 643.39 | 467.14 |

The first implementation performed score reductions and host readbacks per
request. It has been replaced with batched reductions and transfers; the initial
measurements remain as evidence of the overhead that motivated that change.

The batched implementation passes the same 24 independent checks and service
proofs on both GPUs. Maximum selected-score error is 0.00004387 on accurate
Qwen3 A2 and 0.00015402 on Qwen2 M4. Mixed per-request top counts and UTF-8
stream events with temporarily empty text are covered by the local suite.

After batching, top-five reporting measures 184.01 / 326.92 / 544.30 tokens/s
at concurrency 1/2/4, with TTFT 30.84 / 52.24 / 76.44 ms. This recovers much
of the initial overhead, but remains slower than the earlier scoring-off
205.22 / 351.42 / 643.39 tokens/s. Additional score kernels, host transfers and
larger JSON responses are opt-in costs, not claimed to be free.
