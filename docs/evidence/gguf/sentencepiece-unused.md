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

## Byte-level UNUSED pieces (#381)

A Llama3 whole-word lookup exposed another silent mismatch: alphabet plus EOS,
normal `aa`, UNUSED `ab`, and only merge `a a` encoded `ab` as `[64,65]` in dllm
versus reference `[258]`. Qwen2's control case returned `[64,65]` in both.

Byte-level vocabularies now retain unique UNUSED spellings for lookup and merges.
The shared wrapper suppresses those IDs during decoding regardless of special
visibility, preserves the suppression set through serialization, and supplies
it to incremental ByteLevel decoding and logprob presentation. Suppressed pieces
do not flush an incomplete UTF-8 sequence. Structured output with suppressed
pieces remains explicitly rejected; duplicate spellings and UNUSED EOS IDs are
still rejected before loading.

`tests/reference/generate_gguf_bpe_unused.py` freezes 696 independent CPU cases:
two pre-tokenizers, six UNUSED selections, two automatic-special policies,
24 prompt/policy combinations plus five explicit decode sequences per vocabulary.
Coverage includes single bytes, merged pieces, whole-word pieces absent from the
merge table, formerly user-defined pieces, simultaneous suppression and split
UTF-8 bytes. Normal/restored full and batch decoding, exact encoding, incremental
streaming and logprob presentation are checked. This is tokenizer compatibility
evidence only; the model and GPU qualification gates remain open.

## Literal byte-level special text (#382)

For a user-defined `<é>` piece, the raw Rust ByteLevel decoder interpreted the
literal `é` as a byte-alphabet symbol and emitted `<�>`. The reference emits
`<é>`. The shared wrapper now identifies added spellings changed by its backend
decoder, decodes ordinary spans normally, and emits those affected spellings
literally. Hidden controls and UNUSED entries do not split UTF-8 sequences.
Vocabularies without affected spellings retain the direct backend decode path.

`tests/reference/generate_gguf_bpe_literals.py` freezes 160 independent CPU cases:
both byte-level pre-tokenizers, user/control pieces, four Unicode/byte-alphabet
special spellings, repeated and mixed prompts, and explicit partial UTF-8 spans.
Tests compare full/batch decoding before/after serialization and incremental
streaming. For deliberately incomplete UTF-8, reference raw bytes are converted
to text with replacement, matching the public text-decoding contract. Logprob
presentation already preserved literal spellings and remains covered.

## Literal SentencePiece specials (#383)

A user-defined `▁user` at ID 265 encoded correctly but full decoding returned
`user`, and generated decoding returned ` user`; the reference preserves literal
`▁user` in both. SentencePiece now shares literal-span decoding while retaining
its original-first-token whitespace rule. Normal pieces still replace `▁`
with a space. Generated output remains unstripped. Hidden initial controls and
UNUSED pieces still consume the first-token strip opportunity even when invisible.

`tests/reference/generate_sentencepiece_literals.py` freezes 448 CPU cases across
four literal spellings, control/user types, static/positional score paths and
both prefix flags. Each vocabulary includes an UNUSED piece. Prompt and explicit
decode sequences cover leading literal spaces, normal/literal adjacency, hidden
controls, suppression and partial UTF-8. Full/generated/batch decoding and
serialization match the pinned reference; 79 affected CPU tests passed locally.

## CPU grammar qualification (#389, #390)

The earlier structured-output restriction is superseded by explicit RAW generated-byte vocabularies. Actual pinned xgrammar matcher/mask tests cover literal Unicode, byte fragments, UNUSED suppression and positional SentencePiece, including serialization and prefix settings. Hidden pieces stay excluded while explicit EOS IDs terminate valid JSON. Encoding tie-breaking is independent of generated-byte matching. No GPU inference or performance is qualified by these tests.
