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

#376 first rejected UNUSED metadata before loading (bbb38b9), preventing silent
wrong IDs. #377 now implements the independently checked unique-spelling subset.
UNUSED entries participate in encoding, including positional score ties, but
are suppressed in full and generated decoding even with special tokens visible.
An invisible first piece still consumes the one-time leading-space strip.
Serialization preserves suppression. Logprob presentation emits empty text and
empty bytes; IDs and probabilities remain available. Duplicate spellings and
UNUSED explicit stop IDs are rejected. Structured output remains explicitly
unqualified for tokenizers with suppressed pieces.

## Independent CPU oracle

`tests/reference/sentencepiece_oracle.cpp` calls the pinned public vocabulary API
with `vocab_only=true` and `n_gpu_layers=0`; it creates no model context and loads
no tensor weights. It emits exact IDs and byte-hex full/generated decoding for
both special-token visibility settings. Build it against the CPU-only library:

```sh
cmake -S /path/to/llama.cpp -B /tmp/dllm-tokenizer-reference-build \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_MTMD=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dllm-tokenizer-reference-build --target llama --parallel 2
c++ -std=c++17 -O2 tests/reference/sentencepiece_oracle.cpp \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  -L/tmp/dllm-tokenizer-reference-build/bin -lllama \
  -Wl,-rpath,/tmp/dllm-tokenizer-reference-build/bin -o /tmp/dllm-spm-oracle
.venv/bin/python -m tests.reference.generate_sentencepiece_unused /tmp/dllm-spm-oracle
.venv/bin/python -m unittest tests.test_gguf_sentencepiece
```

The frozen fixture records 744 reference calls: six UNUSED selections (including
multiple simultaneous pieces), two score paths and two prefix policies, each
with 22 prompt/special-policy combinations and nine explicit decode sequences.
These cover whitespace, ordinary/special boundaries, byte fallback, Unicode,
NUL, and UNUSED pieces between UTF-8 bytes. Tests compare original and restored
tokenizers against all five reference outputs. The reference runs only when
regenerating the fixture; ordinary CPU CI uses frozen results.

This proves the covered tokenizer semantics, not model quality or GPU execution.
Broader #43 hardware qualification remains open.

## Overlapping special pieces (#378)

A second independent case exposed static-score tokenization choosing a shorter,
leftmost special piece while llama.cpp partitions longer pieces globally first.
With user-defined `ab` and `bcx`, `abcx` encoded as `[262,259,102,123]` instead
of `[263,266]`. Both score paths now share the partition step; static normal
fragments still use Rust BPE. The serialized wrapper preserves this policy even
with automatic space-prefix insertion disabled.

`tests/reference/generate_sentencepiece_specials.py` uses the same CPU oracle to
freeze another 384 cases: control/user-defined specials, static/positional score
paths, both prefix policies, optional automatic EOS, automatic-special toggles,
and repeated/adjacent/overlapping specials plus Unicode and byte fallback.
Tests compare exact IDs, full/generated decoding and batch encoding before and
after serialization. These fixtures exercise unequal-length overlaps; they do
not establish portable ordering for equal-byte-length overlapping specials.

## Equal-length overlap boundary (#379)

The next CPU reference probe confirmed the equal-length caveat is observable,
not hypothetical. Append user-defined specials `abx`, 32 fillers `x00` through
`x31`, and `bxy` to the original tiny vocabulary. Prompt `abxy` without automatic
specials produced `[266,259,124]` in dllm and `[263,299]` in the pinned reference.
With four fillers, both returned `[266,259,124]`. The C++ reference sorts specials
only by byte length using `std::sort`; its equal-length order is unspecified.

Import now rejects distinct equal-byte-length special pieces whose proper
suffix/prefix overlap would make partitioning depend on that order. It retains
non-overlapping length ties, self-overlapping pieces, unequal-byte-length
overlaps, and UNUSED pieces (which are not partitioned as specials). The check
runs before either tokenizer construction path. Sorted prefix searches avoid
quadratic pairwise comparisons within same-length vocabulary groups. Unicode
length is measured in UTF-8 bytes, matching the reference comparator.

## Shared byte-level GGUF partitioning (#380)

The same shorter-leftmost special bug was reproduced for both `qwen2` and
`llama-bpe` GPT-2 pre-tokenizers: byte alphabet plus EOS, `ab` and `bcx` gave
`[257,66,87]` for `abcx`, versus reference `[64,258]`. The shared
`SpecialTokenTokenizer` now partitions specials before calling the original
normal-fragment encoder. SentencePiece keeps its own whitespace/suppression
and positional encoding methods; byte-level decoding remains unchanged.
Equal-length ambiguity rejection applies to both GGUF tokenizer families.

`tests/reference/generate_gguf_bpe_specials.py` freezes 192 additional independent
CPU cases across the two pre-tokenizers, user/control pieces, automatic BOS/EOS
policies, special toggles and overlapping/adjacent/Unicode/byte-fallback prompts.
Original and restored wrappers match exact IDs, decoding and batch behavior.
Separate tests preserve byte-level incremental streaming and logprob presentation.
The frozen matrix runs in CPU CI without building the reference or using GPUs.
