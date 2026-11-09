"""#719 rung 5: shared fleet-chain machinery (4-host full-model cell).

Components, each individually qualified by rungs 1-4a:

- fp8 attention layers: the production CudaCachedGlmAttention composition
  with the SHIPPED blockwise-E4M3-FP8 projections (attention, kv_b, o_proj
  AND the indexer's wq_b/wk) — attention stays fp8 by the rung-4b decision;
  token selection is therefore exact (#668).
- host-resident canonical-q4_0 expert banks (rung-1 packers, rung-2 admission).
- selected-expert streaming (rung 4a: bit-identical to residency): route on
  device, fetch only top-8 selected experts + shared per batch.
- inline paired fp8 oracle: the SAME router selection replayed through the
  shipped-fp8 codes of ONLY the selected experts, read from the NAS
  checkpoint on demand (the batched composition depends on selected experts
  alone, so the oracle's I/O is bounded by the streaming bound itself).

Block composition (glm_moe_dsa, plain pre-norm residual):
  attended = attention(rms_norm(hidden, input_norm))
  residual = hidden + attended
  output   = residual + ffn(rms_norm(residual, post_norm))
"""
import json

import torch
from safetensors import safe_open

from damendaemon.cuda_glm_cached_attention import CudaCachedGlmAttention
from damendaemon.cuda_glm_ffn import _select
from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
from damendaemon.cuda_prepared_projection import CudaPackedProjection
from damendaemon.expert_routes import grouped_routes
from damendaemon.glm_normalization import rms_norm
from damendaemon.glm_q4 import decode_blockwise_fp8
from damendaemon.quantization import quantize_q4

ATTENTION_QUANTIZED = ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj')
ATTENTION_DENSE = ('q_a_layernorm', 'kv_a_layernorm')
INDEXER_QUANTIZED = ('wq_b', 'wk')
INDEXER_DENSE = ('weights_proj', 'k_norm')


def _shard_map(model):
    return json.load(open(f'{model}/model.safetensors.index.json'))['weight_map']


def _load_tensors(model, names):
    """Read exact checkpoint tensors once, grouped by shard."""
    index = _shard_map(model)
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(f'checkpoint misses {missing[:3]}')
    by_shard = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)
    tensors = {}
    for shard, shard_names in by_shard.items():
        with safe_open(f'{model}/{shard}', framework='pt', device='cpu') as handle:
            for name in shard_names:
                tensors[name] = handle.get_tensor(name)
    return tensors, len(by_shard)


def _codes_and_scales(tensors, name, scales_name):
    raw = tensors[name]
    if raw.dtype == torch.uint8:
        codes = raw
    elif raw.dtype == torch.float8_e4m3fn:
        codes = raw.contiguous().view(torch.uint8)
    else:
        raise ValueError(f'unexpected shipped codes dtype {raw.dtype} for {name}')
    scales = tensors[scales_name]
    if scales.dtype != torch.float32:
        raise ValueError(f'unexpected shipped scale dtype {scales.dtype} for {name}')
    return codes, scales


def load_geometry(model):
    """The glm_moe_dsa flat-config geometry this cell exercises."""
    with open(f'{model}/config.json') as reader:
        config = json.load(reader)
    if config.get('model_type') != 'glm_moe_dsa':
        raise ValueError(f"unexpected model_type {config.get('model_type')}")
    return dict(heads=config['num_attention_heads'], nope=config['qk_nope_head_dim'],
                rope=config['qk_rope_head_dim'], value=config['v_head_dim'],
                epsilon=config['rms_norm_eps'], top_k=config['index_topk'],
                rope_base=config['rope_parameters']['rope_theta'],
                dim=config['hidden_size'], layers=config['num_hidden_layers'],
                first_dense=config['first_k_dense_replace'],
                experts_per_tok=config['num_experts_per_tok'],
                route_scale=config['routed_scaling_factor'],
                vocab=config['vocab_size'],
                indexer_types=config['indexer_types'])


def load_attention_tensors(model, layer):
    """Every self_attn tensor of one layer (packed fp8, dense fp32, indexer)."""
    attn = f'model.layers.{layer}.self_attn'
    names = [f'{attn}.{n}.weight' for n in ATTENTION_QUANTIZED + ATTENTION_DENSE]
    names += [f'{attn}.{n}.weight_scale_inv' for n in ATTENTION_QUANTIZED]
    names += [f'{attn}.indexer.{n}.weight' for n in INDEXER_QUANTIZED + INDEXER_DENSE]
    names += [f'{attn}.indexer.{n}.weight_scale_inv' for n in INDEXER_QUANTIZED]
    names += [f'{attn}.indexer.k_norm.bias']
    return _load_tensors(model, names)


def build_attention(tensors, model, layer, device, geometry):
    """The production CudaCachedGlmAttention for one real layer, all-fp8 shipped."""
    attn = f'model.layers.{layer}.self_attn'
    weights = {}
    for name in ATTENTION_QUANTIZED:
        codes, scales = _codes_and_scales(tensors, f'{attn}.{name}.weight',
                                         f'{attn}.{name}.weight_scale_inv')
        weights[f'{name}.weight'] = CudaPackedProjection(
            codes.to(device), scales.to(device), format='e4m3fn_f32', dtype=torch.float32)
    for name in ATTENTION_DENSE:
        weights[f'{name}.weight'] = tensors[f'{attn}.{name}.weight'].float().to(device)
    for name in INDEXER_DENSE:
        weights[f'indexer.{name}.weight'] = tensors[f'{attn}.indexer.{name}.weight'].float().to(device)
    weights['indexer.k_norm.bias'] = tensors[f'{attn}.indexer.k_norm.bias'].float().to(device)
    for name in INDEXER_QUANTIZED:
        codes, scales = _codes_and_scales(tensors, f'{attn}.indexer.{name}.weight',
                                         f'{attn}.indexer.{name}.weight_scale_inv')
        weights[f'indexer.{name}.weight'] = CudaPackedProjection(
            codes.to(device), scales.to(device), format='e4m3fn_f32', dtype=torch.float32)
    return CudaCachedGlmAttention(
        weights, heads=geometry['heads'], nope_dim=geometry['nope'], rope_dim=geometry['rope'],
        value_dim=geometry['value'], index_top_k=geometry['top_k'],
        rope_base=geometry['rope_base'], epsilon=geometry['epsilon'],
        indexer_type='full' if geometry['indexer_types'][layer] == 'full' else 'shared',
        max_workspace_bytes=512 * 1024**2, max_output_bytes=256 * 1024**2)


def load_block_norms(model, layer):
    prefix = f'model.layers.{layer}'
    names = [f'{prefix}.input_layernorm.weight', f'{prefix}.post_attention_layernorm.weight']
    tensors, shards = _load_tensors(model, names)
    return (tensors[names[0]].float(), tensors[names[1]].float()), shards


def load_dense_ffn_tensors(model, layer):
    """Dense-layer MLP (layers < first_k_dense_replace) — shipped fp8, stays fp8."""
    prefix = f'model.layers.{layer}.mlp'
    names = [f'{prefix}.{part}_proj.weight' for part in ('gate', 'up', 'down')]
    names += [f'{prefix}.{part}_proj.weight_scale_inv' for part in ('gate', 'up', 'down')]
    return _load_tensors(model, names)


def resident_fp8_expert(tensors, prefix, device):
    """One expert as a resident fp8 CudaPackedExpert from raw checkpoint codes."""
    matrices = []
    for part in ('gate', 'up', 'down'):
        codes, scales = _codes_and_scales(tensors, f'{prefix}.{part}_proj.weight',
                                         f'{prefix}.{part}_proj.weight_scale_inv')
        matrices.append(CudaPackedMatrix(codes.to(device), scales.to(device), 'e4m3fn_f32'))
    return CudaPackedExpert(*matrices)


def load_moe_tensors(model, layer):
    """Router, bias and every expert/shared matrix of one real MoE layer."""
    prefix = f'model.layers.{layer}.mlp'
    index = _shard_map(model)
    names = [f'{prefix}.gate.weight', f'{prefix}.gate.e_score_correction_bias']
    experts = 0
    while f'{prefix}.experts.{experts}.gate_proj.weight_scale_inv' in index:
        experts += 1
    if experts < 2:
        raise ValueError(f'layer {layer} is not a sparse-MoE layer')
    for expert in list(range(experts)) + ['shared_experts']:
        owner = f'{prefix}.experts.{expert}' if expert != 'shared_experts' else f'{prefix}.shared_experts'
        for part in ('gate', 'up', 'down'):
            names.append(f'{owner}.{part}_proj.weight')
            names.append(f'{owner}.{part}_proj.weight_scale_inv')
    tensors, shards = _load_tensors(model, names)
    return tensors, experts, shards


def pack_q4_bank(moe_tensors, experts, layer, device, batch_rows=8):
    """Pack the layer's full expert bank to canonical q4_0, resident in HOST RAM.

    Decoding/quantization run on the executor device one expert at a time
    (bounded transient VRAM), then the packed q4 bytes move to host RAM — the
    rung-2 measured pack path, extended per-expert so the bank is host-resident.
    """
    prefix = f'model.layers.{layer}.mlp'
    host_bank = {}
    packed_bytes = 0
    for expert in range(experts):
        owner = f'{prefix}.experts.{expert}'
        triple = []
        for part in ('gate', 'up', 'down'):
            codes, scales = _codes_and_scales(moe_tensors, f'{owner}.{part}_proj.weight',
                                             f'{owner}.{part}_proj.weight_scale_inv')
            truth = decode_blockwise_fp8(codes.to(device), scales.to(device))
            weight, block_scales = quantize_q4(truth)
            triple.append((weight.to('cpu'), block_scales.to('cpu')))
            packed_bytes += weight.numel() + block_scales.numel() * 2
            del truth, weight, block_scales
        host_bank[expert] = triple
    shared = []
    for part in ('gate', 'up', 'down'):
        codes, scales = _codes_and_scales(moe_tensors, f'{prefix}.shared_experts.{part}_proj.weight',
                                         f'{prefix}.shared_experts.{part}_proj.weight_scale_inv')
        truth = decode_blockwise_fp8(codes.to(device), scales.to(device))
        weight, block_scales = quantize_q4(truth)
        shared.append((weight.to('cpu'), block_scales.to('cpu')))
        packed_bytes += weight.numel() + block_scales.numel() * 2
        del truth, weight, block_scales
    torch.cuda.empty_cache()
    router = moe_tensors[f'{prefix}.gate.weight'].float()
    bias = moe_tensors[f'{prefix}.gate.e_score_correction_bias'].float()
    return host_bank, shared, router, bias, packed_bytes


def _q4_expert(triple, device):
    return CudaPackedExpert(*[CudaPackedMatrix(w.to(device), s.to(device), 'q4_0_f16')
                              for w, s in triple])


def streamed_composition(rows, ids, coefficients, bank_get, shared_expert, experts):
    """Rung-4a streamed composition: identical accumulation order for any bank.

    rows [tokens, dim] fp32 on device; ids/coefficients from _select. bank_get
    maps an expert id to (resident CudaPackedExpert on device, bytes moved).
    Each selected expert is visited exactly once (stable expert groups).
    Returns (output [tokens, dim], streamed_bytes).
    """
    streamed_bytes = 0
    output = torch.zeros_like(rows)
    flat_coefficients = coefficients.reshape(-1)
    for expert_id, tokens_for_expert, positions in grouped_routes(ids, experts):
        expert, moved = bank_get(expert_id)
        streamed_bytes += moved
        source = rows.index_select(0, tokens_for_expert)
        value = expert.forward_batched(source)
        scaled = value.float() * flat_coefficients[positions].reshape(-1, 1)
        output.index_add_(0, tokens_for_expert, scaled)
        del expert, source, value
    if shared_expert is not None:
        shared_out = shared_expert.forward_batched(rows)
        output.add_(shared_out.float())
        del shared_out
    return output, streamed_bytes


def _q4_bank_get_factory(host_bank, device):
    """Stream one host-resident q4 expert to the device on demand."""

    def bank_get(expert_id):
        triple = host_bank[expert_id]
        moved = sum(w.numel() + s.numel() * 2 for w, s in triple)
        return _q4_expert(triple, device), moved

    return bank_get


def _fp8_bank_get_factory(model, layer, device):
    """On-demand fp8 oracle: read ONLY the selected experts' shipped codes."""
    prefix = f'model.layers.{layer}.mlp'

    def bank_get(expert_id):
        owner = f'{prefix}.experts.{expert_id}'
        names = [f'{owner}.{part}_proj.{suffix}' for part in ('gate', 'up', 'down')
                 for suffix in ('weight', 'weight_scale_inv')]
        tensors, _ = _load_tensors(model, names)
        expert = resident_fp8_expert(tensors, owner, device)
        moved = sum(t.numel() if t.dtype == torch.uint8 else t.numel() * 4
                    for t in tensors.values())
        return expert, moved

    return bank_get


def moe_layer_forward(hidden, model, layer, host_bank, host_shared, router, bias,
                      device, geometry, oracle=True):
    """One MoE layer's FFN: streamed q4 chain output + inline paired fp8 oracle.

    Returns (y_q4 [b, s, dim], gate dict) — the chain continues on y_q4 while
    the gate records q4-vs-fp8 parity on the ACTUAL chain activations with the
    SAME router selection (fp32 exact, #668) and identical accumulation order.
    """
    rows = hidden.reshape(-1, hidden.shape[-1])
    experts = len(host_bank)
    logits = torch.nn.functional.linear(rows, router.to(device))
    coefficients, ids = _select(logits, bias.to(device), geometry['experts_per_tok'],
                                geometry['route_scale'])
    bank_get = _q4_bank_get_factory(host_bank, device)
    shared_q4 = _q4_expert(host_shared, device)
    y_q4, streamed_bytes = streamed_composition(rows, ids, coefficients, bank_get,
                                                shared_q4, experts)
    gate = dict(layer=layer, selected=int(ids.numel()),
                distinct_selected=len(set(ids.reshape(-1).tolist())),
                streamed_bytes=streamed_bytes,
                host_bank_bytes=geometry['_bank_bytes'])
    gate['streaming_bounded'] = 0 < streamed_bytes < geometry['_bank_bytes']
    if oracle:
        fp8_bank_get = _fp8_bank_get_factory(model, layer, device)
        fp8_shared_names = [f'model.layers.{layer}.mlp.shared_experts.{part}_proj.{suffix}'
                            for part in ('gate', 'up', 'down')
                            for suffix in ('weight', 'weight_scale_inv')]
        shared_tensors, _ = _load_tensors(model, fp8_shared_names)
        fp8_shared = resident_fp8_expert(shared_tensors, f'model.layers.{layer}.mlp.shared_experts', device)
        y_oracle, _ = streamed_composition(rows, ids, coefficients, fp8_bank_get,
                                           fp8_shared, experts)
        delta = y_q4 - y_oracle
        rel = float((delta.norm() / y_oracle.norm()).item())
        max_abs = float(delta.abs().max().item())
        absmax = float(y_oracle.abs().max().item())
        gate.update(rel_rms=rel, max_abs=max_abs, oracle_absmax=absmax,
                    max_abs_of_absmax=max_abs / absmax)
        gate['parity'] = rel <= 0.15 and max_abs <= 0.15 * absmax
        gate['finite'] = bool(torch.isfinite(y_q4).all() and torch.isfinite(y_oracle).all())
        del y_oracle, fp8_shared
    del logits, coefficients, ids, shared_q4
    torch.cuda.empty_cache()
    return y_q4.reshape(hidden.shape), gate
