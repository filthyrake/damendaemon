# SentencePiece UNUSED token compatibility boundary

Issue #376, parent #43. CPU vocabulary-only reproduction; no inference, weights,
GPU qualification or running service access.

Reference: the existing unmodified llama.cpp checkout at
`836d57176dc699a726c55418e4f96b8ca628e1bf`. CMake 4.1.2 was installed into a
private temporary tooling directory. `llama-tokenize` was built outside the
checkout with `GGML_METAL=OFF`, `GGML_CUDA=OFF`, `GGML_OPENMP=OFF`, two CPU build
jobs, and bounded configure/build timeouts. No upstream files changed.

The fixture uses `metadata()` from `tests/test_gguf_sentencepiece.py`, written
with the GGUF writer to a vocabulary-only Llama file (context 128, width 32,
feed-forward width 64, one block, four query heads/two KV heads, epsilon 1e-6).
The baseline vocabulary is unchanged. The second file changes only token 260,
`a`, from NORMAL (1) to UNUSED (5). No weight tensors are written.

For `llama-tokenize -m <fixture.gguf> -p a --ids`:

| Input | Pinned reference | dllm before correction |
|---|---|---|
| Baseline vocabulary, prompt `a` | `[1, 263]` | `[1, 263]` |
| UNUSED `a`, prompt `a` | `[1, 263]` | `[1, 259, 100]` |

The reference tool explicitly sets `vocab_only=true`. Its SPM implementation
considers the token map for merges/resegmentation without filtering UNUSED;
its piece decoder suppresses unsupported attributes such as UNUSED. dllm's
static-score vocabulary previously dropped these entries, while its positional
path already rejected them. The same accepted file could therefore encode a
different prompt without a validation error.

Both dllm paths now reject UNUSED metadata before tokenizer/checkpoint loading.
The regression retains the ordinary `[1,263]` control and covers UNUSED single
characters, merge pieces and user-defined entries. Supporting these pieces
requires an independently checked encode/decode/roundtrip contract, not simply
removing the guard or marking them NORMAL. This remains a software capability
boundary under #43; no idle GPU is required to investigate it further.
