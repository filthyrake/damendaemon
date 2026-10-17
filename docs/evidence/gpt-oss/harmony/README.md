# Text Harmony qualification

Tracking: [#39](https://github.com/filthyrake/damendaemon/issues/39).
Checkpoint and core numerical qualification are recorded in the parent folder.

- `harmony-format-generated-reference.json`: 19 CPU prompt/parser comparisons.
  Reference: Transformers 5.18 for the checkpoint's exact prompt template,
  official OpenAI Harmony 0.0.8 (Rust implementation) for generated token streams.
- `harmony-native-chat-a2.json`: actual independent native A2 generation,
  including token IDs subsequently checked by the official parser.
- `harmony-history-service-a2.json`: real 512-context/two-slot HTTP checks,
  including Unicode, reasoning/answer separation, token caps, stop strings,
  response-history round trips, streaming, cancellation, concurrency and swaps.

Reproduction: `scripts/verify_harmony_format.py`, `scripts/probe_harmony_chat.py`,
`scripts/verify_harmony_service.py`. The official parser is a validation-only
reference, not a new execution dependency. Format source:
[OpenAI Harmony](https://github.com/openai/harmony), Apache-2.0; implementation
is independent and no upstream parser source was copied.

Text chat only: unsupported function handoffs, JSON-constrained chat and chat
logprobs remain explicit errors. Passing these protocol checks does not imply
broad model-quality, tool-use, long-context or cross-engine performance parity.
