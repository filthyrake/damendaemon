"""Shared real-GLM-5.3 q4/fp8 bank construction for the #719 chain cells."""
import json

import torch
from safetensors import safe_open

from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward
from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
from damendaemon.glm_q4 import decode_blockwise_fp8
from damendaemon.quantization import quantize_q4


def load_ffn_tensors(model, layer):
    """Router, bias and every expert/shared matrix of one real MoE layer."""
    index = json.load(open(f'{model}/model.safetensors.index.json'))['weight_map']
    wanted = [f'model.layers.{layer}.mlp.gate.weight',
              f'model.layers.{layer}.mlp.gate.e_score_correction_bias']
    experts = 0
    while f'model.layers.{layer}.mlp.experts.{experts}.gate_proj.weight_scale_inv' in index:
        experts += 1
    if experts < 2:
        raise ValueError(f'layer {layer} is not a sparse-MoE layer')
    prefixes = [f'model.layers.{layer}.mlp.experts.{i}' for i in range(experts)]
    prefixes.append(f'model.layers.{layer}.mlp.shared_experts')
    for prefix in prefixes:
        for part in ('gate', 'up', 'down'):
            wanted.append(f'{prefix}.{part}_proj.weight')
            wanted.append(f'{prefix}.{part}_proj.weight_scale_inv')
    missing = [name for name in wanted if name not in index]
    if missing:
        raise ValueError(f'checkpoint misses {missing[:3]}')
    by_shard = {}
    for name in wanted:
        by_shard.setdefault(index[name], []).append(name)
    tensors = {}
    for shard, names in by_shard.items():
        with safe_open(f'{model}/{shard}', framework='pt', device='cpu') as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    return tensors, experts


def _packed_matrix(tensors, prefix, part, device, q4):
    raw = tensors[f'{prefix}.{part}_proj.weight']
    if raw.dtype == torch.uint8:
        codes = raw
    elif raw.dtype == torch.float8_e4m3fn:
        codes = raw.contiguous().view(torch.uint8)
    else:
        raise ValueError(f'unexpected shipped codes dtype {raw.dtype}')
    scales = tensors[f'{prefix}.{part}_proj.weight_scale_inv']
    if scales.dtype != torch.float32:
        raise ValueError(f'unexpected shipped scale dtype {scales.dtype}')
    codes, scales = codes.to(device), scales.to(device)
    if not q4:
        return CudaPackedMatrix(codes, scales, 'e4m3fn_f32')
    weight, block_scales = quantize_q4(decode_blockwise_fp8(codes, scales))
    return CudaPackedMatrix(weight, block_scales, 'q4_0_f16')


def build_ffn(tensors, experts, layer, device, *, q4, top_k=8):
    """One real layer's FFN: q4_0_f16 bank or the shipped fp8 oracle."""
    bank = [CudaPackedExpert(*[_packed_matrix(tensors, f'model.layers.{layer}.mlp.experts.{expert}',
                                               part, device, q4)
                                for part in ('gate', 'up', 'down')])
            for expert in range(experts)]
    shared = CudaPackedExpert(*[_packed_matrix(tensors, f'model.layers.{layer}.mlp.shared_experts',
                                               part, device, q4)
                                for part in ('gate', 'up', 'down')])
    router = tensors[f'model.layers.{layer}.mlp.gate.weight'].float().to(device)
    bias = tensors[f'model.layers.{layer}.mlp.gate.e_score_correction_bias'].float().to(device)
    return CudaPackedGlmFeedForward(bank, router=router, correction_bias=bias, shared=shared,
                                    top_k=top_k, composition='batched')
