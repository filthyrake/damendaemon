"""#719 rung 1: q4_0 packing + drift record on real official GLM-5.3 weights.

Cell A (weight-level): read representative tensors from the staged official
checkpoint (NAS), dequantize the shipped blockwise-E4M3 FP8 to FP32 (the
truth the model actually ships), pack with the repo's canonical
`quantization.py:quantize_q4` (uint8 paired nibbles + FP16 per-32 scales),
unpack, and record paired-delta drift. Idempotency gate: re-packing the
unpacked values must reproduce the packed bytes exactly (the group-absmax
element round-trips by construction).

Cell B (projection): the same packing decisions applied to the full
contract (tensor_specs) — per-stage q4 totals at the #56 v5 boundary-legal
ranges — so the fleet plan can be re-derived at q4 scale.

Packing decisions mirrored from quantization.prepare(): every 2-D matrix is
packed EXCEPT embeddings, routers (mlp.gate.*) and shared-expert combine
gates; those stay high precision (selection semantics pinned exact, #668).
"""
import json, math, sys, time
sys.path.insert(0, '/tmp/damendaemon-g56')
import torch
from safetensors import safe_open

MODEL = '/mnt/nas-models/models/GLM-5.3'
OUT = '/tmp/glm56-3host/rung1-q4'

from damendaemon.quantization import quantize_q4, dequantize_q4
from damendaemon.glm_checkpoint import tensor_specs

torch.set_num_threads(16)
index = json.load(open(f'{MODEL}/model.safetensors.index.json'))['weight_map']


def load_pair(name):
    """Return the FP32 truth of a blockwise-E4M3 FP8 tensor + source bytes.

    lm_head ships unquantized (no _scale_inv companion; the contract's
    'floating' storage) — load it directly and float() it.
    """
    shard = f'{MODEL}/{index[name]}'
    with safe_open(shard, framework='pt', device='cpu') as f:
        val = f.get_tensor(name)
        names = set(f.keys())
        if f'{name}_scale_inv' not in names:
            return val.float(), val.nbytes
        scale = f.get_tensor(f'{name}_scale_inv')
    src_bytes = val.nbytes + scale.nbytes
    # blockwise scale: weight [R,W] carries scale [R/br, W/bw] (128x128 blocks),
    # so scale_inv must expand over BOTH axes. E4M3 stored value * scale_inv.
    vs = val.shape
    ss = scale.shape
    if len(vs) != 2 or len(ss) != 2:
        raise ValueError(f'unhandled rank for {name}: {vs} / {ss}')
    # 128-element blocks on both axes with a PARTIAL tail block
    # (e.g. kv_a_proj 576 rows -> 5 scale rows: 4x128 + 64)
    BLOCK = 128
    if -(-vs[0] // BLOCK) != ss[0] or -(-vs[1] // BLOCK) != ss[1]:
        raise ValueError(f'unexpected 128-block grid for {name}: {ss} vs {vs}')
    rows_idx = torch.arange(vs[0]) // BLOCK
    cols_idx = torch.arange(vs[1]) // BLOCK
    if rows_idx.max() >= ss[0] or cols_idx.max() >= ss[1]:
        raise ValueError(f'block grid mismatch for {name}: {ss} vs {vs}')
    expanded = scale[rows_idx][:, cols_idx]
    # E4M3FN byte -> float
    f8 = val.to(torch.float8_e4m3fn).float() if hasattr(torch, 'float8_e4m3fn') else None
    if f8 is None:
        raise RuntimeError('no float8_e4m3fn support')
    truth = f8 * expanded.float()
    return truth, src_bytes


PACKED_PICKS = [
    'model.layers.0.self_attn.q_a_proj.weight',
    'model.layers.0.self_attn.indexer.wk.weight',
    'model.layers.6.self_attn.kv_a_proj_with_mqa.weight',
    'model.layers.6.self_attn.q_b_proj.weight',
    'model.layers.6.mlp.experts.0.gate_proj.weight',
    'model.layers.6.mlp.experts.255.down_proj.weight',
    'model.layers.6.mlp.shared_experts.up_proj.weight',
    'lm_head.weight',
]
UNPACKED_CLASSES = [
    'model.embed_tokens.weight',
    'model.layers.6.mlp.gate.weight',
    'model.layers.6.mlp.gate.e_score_correction_bias',
]

rows = []
for name in PACKED_PICKS:
    t0 = time.time()
    truth, src_bytes = load_pair(name)
    data, scales = quantize_q4(truth)
    back = dequantize_q4(data, scales)
    delta = back - truth
    rel_rms = (delta.norm() / truth.norm()).item()
    max_abs = delta.abs().max().item()
    denom = truth.abs().max().item()
    # idempotency: re-pack the unpacked values
    data2, scales2 = quantize_q4(back)
    idem = bool(torch.equal(data, data2) and torch.equal(scales, scales2))
    packed_bytes = data.nbytes + scales.nbytes
    row = dict(name=name, shape=list(truth.shape), elements=truth.numel(),
               source_bytes=src_bytes, packed_bytes=packed_bytes,
               ratio=round(packed_bytes / src_bytes, 4),
               rel_rms=rel_rms, max_abs=max_abs,
               max_abs_rel_to_absmax=round(max_abs / denom, 6),
               idempotent=idem, seconds=round(time.time() - t0, 1))
    rows.append(row)
    print(f'{name}: rel_rms={rel_rms:.5f} max_abs={max_abs:.3e} '
          f'({max_abs / denom:.4f} of absmax) ratio={row["ratio"]:.3f} idem={idem}', flush=True)

# unpacked classes: record shapes/sizes only (decision, not measurement)
unpacked = []
for name in UNPACKED_CLASSES:
    with safe_open(f'{MODEL}/{index[name]}', framework='pt', device='cpu') as f:
        t = f.get_tensor(name)
    unpacked.append(dict(name=name, shape=list(t.shape), dtype=str(t.dtype),
                         bytes=t.nbytes, decision='stays high precision (selection/embed semantics)'))
    del t

# ---- cell B: full-contract q4 projection at the v5 stage boundaries ----
fixture = json.load(open('/tmp/damendaemon-g56/tests/fixtures/glm53_contract.json'))
config = fixture['config']
specs = tensor_specs(config)
layers = config['num_hidden_layers']
PACK_BYTES_PER_ELEM = 0.5 + 2.0 / 32  # uint8 nibbles + FP16 scale per 32


def projected_q4_bytes(name, spec):
    e = math.prod(spec.shape)
    if spec.storage == 'fp8' and spec.shape and len(spec.shape) == 2:
        return e * PACK_BYTES_PER_ELEM, 'q4_0'
    # floating 2-D: packed too, except embed/routers/shared-expert gates (qwen
    # convention mirrored); small enough that counting them unpacked is the
    # conservative choice for budget planning
    base = e * 4  # floating retained FP32
    if name == 'model.embed_tokens.weight' or '.mlp.gate.' in name:
        return base, 'unpacked-fp32'
    if len(spec.shape) == 2 and name != 'lm_head.weight':
        return e * PACK_BYTES_PER_ELEM, 'q4_0(floating)'
    return base, 'unpacked-fp32'


stage = [0] * (layers + 2)
q4_total = 0
for name, spec in specs.items():
    if name == 'model.embed_tokens.weight':
        idx = 0
    elif name.startswith('model.layers.'):
        idx = 1 + int(name.split('.')[2])
    else:
        idx = layers + 1
    b, cls = projected_q4_bytes(name, spec)
    stage[idx] += b
    q4_total += b

GIB = 2 ** 30
RANGES = [(0, 62), (62, 70), (70, 78)]
HOSTS = ['labnode3', 'rocky10-desktop', '3090box']
per_stage = []
for (a, b), host in zip(RANGES, HOSTS):
    n = sum(stage[a + 1:b + 1])
    if a == 0:
        n += stage[0]
    if b == layers:
        n += stage[layers + 1]
    per_stage.append(dict(host=host, layers=[a, b], q4_gib=round(n / GIB, 2)))

report = dict(
    scope='glm53_q4_fleet_719_rung1',
    date='2026-11-08',
    official_revision='ba32f8c19c8738e2c57c71798c601d477a7bf844',
    source='NAS original /mnt/nas-models/models/GLM-5.3',
    packing='quantization.quantize_q4 (uint8 paired nibbles + FP16 per-32 scales)',
    truth='blockwise-E4M3-FP8 x scale_inv dequantized to FP32 (shipped values)',
    packed_tensor_rows=rows,
    unpacked_classes=unpacked,
    projection=dict(
        fp8_retained_gib=round(sum(math.prod(s.shape) * (1 if s.storage == 'fp8' else 4) for s in specs.values()) / GIB, 2),
        q4_projected_total_gib=round(q4_total / GIB, 2),
        stage_ranges=[list(r) for r in RANGES],
        per_stage=per_stage,
    ),
)
import os
os.makedirs(OUT, exist_ok=True)
json.dump(report, open(f'{OUT}/report.json', 'w'), indent=1, sort_keys=True)
print('PROJECTION total q4 %.2f GiB | per-stage: %s' % (
    q4_total / GIB, ['%.1f' % (s['q4_gib']) for s in per_stage]))
print('RUNG1_CELL_DONE')
