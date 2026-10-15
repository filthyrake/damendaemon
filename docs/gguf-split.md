# Split GGUF import

Pass the first standard numbered shard, for example
`model-00001-of-00004.gguf`, anywhere a supported GGUF model path is accepted.
All numbered siblings must be present in the same directory. Numbered symlinks
to opaque cache blobs are supported: the importer preserves the logical sibling
namespace while checking each target file identity. Single-file cache symlinks
retain their previous resolution behavior.
CLI selection, runtime registration, and model-manifest export/reload preserve
the same numbered first-shard path. Exported manifests therefore retain the
snapshot directory containing sibling links, rather than an opaque blob path.

For a Hugging Face repository, select the first shard with `--filename`. The
downloader checks that all 2–64 named siblings exist before downloading any
weights, then fetches only that family at the resolved immutable revision.
Missing shards, a non-first selection, or out-of-range counts fail explicitly.
A failed transfer may leave individual files in the Hugging Face cache; it never
returns a successful model selection until every requested file is fetched.

The importer accepts 2–64 shards and at most 100,000 tensors, with a shared
256 MiB aggregate header budget. It validates every shard's index, count and
total tensor count, rejects duplicate tensors and conflicting metadata, and
permits the upstream writer's metadata-only first shard. The first shard owns
the model and tokenizer metadata; later shards may repeat matching metadata.
Alignment is local to each file.

Tensor ownership follows the validated inventory. Fused Q/K/V and gate/up
projections may span shards; packed values, mixed-format layout and tied heads
retain the single-file import policy. Every file participates in metadata-cache
invalidation and the identity checks before and after payload loading. All
payload handles close on success or failure, and cached plans retain no mapped
file views. These checks detect ordinary file replacement/modification; they
are not an authenticated content hash or protection against a hostile writer.

This extends only the existing dense Qwen2/Qwen3/Llama import and its supported
tokenizers and quantization formats. It does not add IQ formats, model families,
GPU kernels or new precision policies. Split fixtures written by the optional
MIT-licensed `gguf` writer match single-file weights and tokenizer metadata in
CPU tests, including K-quants and cross-shard mixed projections. Real split
checkpoint GPU/HTTP qualification remains open in issue #43.

Validation: `python -m unittest tests.test_gguf_split` plus the existing GGUF
regressions. Negative controls cover missing/misnumbered shards, aggregate
bounds, conflicting metadata, duplicate tensors, later-shard cache changes,
changes during payload loading and partial-open cleanup.
