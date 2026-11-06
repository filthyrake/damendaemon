# GLM-5.3 official payload prepare (#56, rung 3)

2026-11-06, labnode3 (`dsv4f-venv`, CPU torch) against the staged official
checkpoint `models/GLM-5.3` (PR #715 admission chain). Cell A: single-group
stage materialization via the production `load_reference_stage` plus
byte/value spot-checks against independent `safetensors` reads — no payload
shortcuts, every check reads the raw tensor from its shard.

## Stage plan discipline (new finding)

GLM-5.3's sparse indexers are grouped: 21 'full' indexer layers (0,1,2 then
every 4th: 6,10,…,74) each head a group followed by 3 'shared' layers.
`validate_range` requires stage boundaries to land on 'full' layers — so
single-layer stages exist only at group heads, and the [74,78) final stage
carries 4 MoE layers + norm + head. Plan ranges must respect this.

## Cell A results (26/26 checks byte/value-exact)

- **[0,1) dense + full-indexer MLA**: 10/10 — q_a_proj fp8-bytes+scales,
  q_a_layernorm values, o_proj fp8+scales, indexer wk fp8 + k_norm.bias
  values, dense-MLP down_proj fp8+scales, embed_tokens values. 13.0 s load,
  4.4 GiB RSS (weights 5.69 GiB per the #714 sweep).
- **[6,10) sparse-MoE group**: 11/11 — kv_a/q_b/kv_b fp8, experts[0].gate
  fp8+scales, experts[255].down fp8, **shared expert up fp8+scales** (the
  cell's one printed FAIL compared `shared.gate` against `up_proj` — a
  check bug, re-verified with the correct `shared.up` attribute: both PASS,
  see `shared-expert-corrected-check.log`), router + e_score_correction_bias
  values, layer-9 (group tail) experts[7].up fp8. 85.5 s load, 37.4 GiB RSS.
- **[74,78) final group + head/norm**: 5/5 — layer-77 kv_b + experts[7].up
  fp8, router values, `lm_head.weight` (154880×6144) and `model.norm.weight`
  values. 95.2 s load; peak cell RSS 56.3 GiB.

Loader budgets exercised (fixture-scale defaults insufficient, passed as
caller parameters per the #712/#713 discipline): `projection_workspace_bytes`
64 MiB (expert fp8 unpack needs ~9.5 MB/matrix > 8 MiB default),
`max_workspace_bytes` 128 MiB (>= projection + 128×(hidden+256 experts)),
per-stage `max_weight_bytes` 8/48/48 GiB. No code change required — the
admission chain from #714/#715 holds under payload reads.

Identity fingerprint (whole-dir model_id, 760 GiB bound) running; value lands
in the topology cell evidence.

## Files

- `cell-a.log` — full run log; `glm56_payload_cell_a.py` — the cell;
  `shared-expert-corrected-check.log` — the corrected shared-expert check.
