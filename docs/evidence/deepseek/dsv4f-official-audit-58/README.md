# #58 — Official DeepSeek-V4-Flash-0731 weights: pinned-audit qualification (CPU, no execution)

**First physical-qualification increment for #58**: the official release's
weights are on the NAS and the repo's pinned checkpoint contract audits
them clean — names, shapes and storage classes — without reading a single
weight value or claiming execution.

## The release

- `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT), HF revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, 74 files,
  166,898,661,074 bytes — downloaded to
  `/mnt/nas-models/models/DeepSeek-V4-Flash-0731` and **byte-count
  verified against the HF API listing** (74/74 files, exact total).
- Architecture (from the official config): 43 layers, 4096 hidden,
  256 routed experts + 1 shared, 6/tok, FP4 experts + FP8 (E4M3) dense
  weights with E8M0 128×128 block scales, hyper-connections (hc_mult 4),
  lightning indexer (64 heads, top-512), compressed attention
  (ratios 4/128 alternating), 3 hash layers, 1M context, o_lora/q_lora
  low-rank attention groups.

## What the pre-release contract got wrong — and the amendment

The contract was pinned to a pre-release tensor index
(`60d8d707…`, 44-ratio schedule). The official release differs in exactly
one structural way:

- `compress_ratios` carries **46** entries: 43 main layers + **3 dspark
  draft blocks** (`dspark_target_layer_ids`, the DeepSeek Spark
  speculative chain — each `mtp.N` block additionally carries
  `confidence_head` and `markov_head` tensors). The contract accepted only
  `layers + num_nextn_predict_layers` = 44 (`num_nextn_predict_layers`
  is 1 in the official config — it does not describe the dspark chain).
- **The main-model inventory is otherwise byte-identical to the
  prediction**: 67,612 tensors, the same name fingerprint
  (`4836d4583cd09fea…`) — the fixture's pinned count and sha256 carry
  over unchanged, which is the strongest possible confirmation the
  pre-release transcription matched the official layout.

Amendment (`damendaemon/deepseek_checkpoint.py`): the draft-block count
derives from `dspark_target_layer_ids` (validated: in-range layer ids,
bounded; `dspark_block_size`/`dspark_markov_rank`/`dspark_noise_token_id`
geometry validated) with the `num_nextn_predict_layers` fallback for
pre-dspark layouts. The docstring/fixture now pin the OFFICIAL revision.
MTP/dspark tensors remain explicitly excluded and reported (4,705),
never validated.

## Audit results against the real checkpoint (labnode3, NAS read-only)

| mode | result |
|---|---|
| `--names-only` (index only, no shard opens) | PASS — 67,612 main tensors exact, 4,705 mtp excluded |
| header contract (safetensors headers: every shape + storage class) | PASS — same counts; zero mismatches |

Storage classes verified: BF16 (floating), F8_E4M3 (dense fp8),
fp4-packed routed experts (I8/U8 rows), F8_E8M0 scales, I32
(`gate.tid2eid` on the 3 hash layers). No weight values were read;
`weight_values_read: false`; `model_execution_qualified: false` — this
is a metadata qualification, not an execution claim.

## Artifacts

- `names-only.json` / `header-contract.json` — the two audit reports,
  produced by `scripts/audit_deepseek_checkpoint` against
  `/mnt/nas-models/models/DeepSeek-V4-Flash-0731` on labnode3.
- `tests/fixtures/deepseek_v4_flash_contract.json` — re-pinned to the
  official config + revision; main fingerprint unchanged from the
  pre-release pin (byte-identical main inventory).
- `tests/test_deepseek_checkpoint.py` — 8 tests, including the dspark
  schedule derivation (tiny fixtures: target-driven tails, out-of-range
  targets, bad dspark geometry, pre-dspark fallback) and the official
  46-ratio/3-block pin.

## Honest scope + next steps (not claimed here)

- No weight VALUES were validated, no model was built, nothing ran on a
  GPU — the pinned contract is metadata-only by design.
- The dspark/MTP blocks (4,705 tensors, confidence/markov heads) are
  excluded and unqualified — serving them is future work under #58.
- Next physical steps: byte-budget planning against the REAL packed
  geometry (155 GiB; FP8 dense + FP4 experts — the existing packed
  planner decides the fleet split), promoted/packed prepare of a host
  scratch copy (the NAS stays read-only), then the first guarded
  executor cells.
