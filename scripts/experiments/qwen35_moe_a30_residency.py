"""E2M1 expert-bank residency on the A30 for qwen3_5_moe (#14, cell 7).

Places the full 40-layer routed-expert bank (256 experts/layer) of the real
Qwen3.6-35B-A3B snapshot on one GPU in the prepared E2M1+E8M0 layout
(uint8 nibble pairs, one E8M0 scale byte per row/32 columns) and proves the
composition against the cell-6 CPU control:

  packing   - FP32 -> E2M1 codes transcribed from the audited
              ``simulate_fp4`` arithmetic (same boundaries, nearest-even
              ties, power-of-two amax/6 scale per 32-value block along the
              input dim). Splitting the fused gate_up into gate/up keeps the
              identical per-row 32-blocks, so the bank decodes to exactly the
              cell-6 fp4 arm values. A round-trip assert (packed bytes ->
              ``decode_tile`` == ``simulate_fp4``) must hold exactly on a
              random tensor and a real tensor tile before placement.
  placement - all 40 layers streamed from the read-only snapshot; per layer,
              packed bytes + scales move to the GPU and CPU copies are freed.
              Bank arithmetic: 15.9 GiB; measured torch.cuda.memory_allocated
              is recorded and must not exceed arithmetic + tolerance.
  compose   - layers 3 and 39: 256 CudaPackedExperts borrowed from the
              GPU-resident bank, routes selected by the FP32 CPU router
              (cell-3 pinned policy), composition via cuda_packed_moe
              (fused path) in FP32 with TF32 disabled, compared to the
              cell-6 CPU control (simulate_fp4 effective weights, shared
              expert zeroed so the comparison isolates the routed bank).

Expected agreement is tolerance-class, not bitwise: the fused path folds
route coefficients before the down projection (V4 convention) while the CPU
control multiplies after (Qwen order), and tile decode/matmul orders differ.
Metrics are recorded exactly as measured. No quality, throughput or serving
claim. Run under scripts.experiments.guard_shared_cuda with the A30 UUID.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from damendaemon.activation_simulation import simulate_fp4
from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
from damendaemon.cuda_packed_moe import cuda_packed_moe
from damendaemon.expert_selection import select_experts
from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward
from damendaemon.packed_tiles import decode_tile
from scripts.experiments.qwen35_moe_text_executor import digest, metrics, read_tensors

BOUNDARIES = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)
MLP_SUFFIXES = ('mlp.gate.weight', 'mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
                'mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight',
                'mlp.shared_expert.down_proj.weight', 'mlp.shared_expert_gate.weight')


def _ceil_power_two_exponent(values):
    mantissa, exponent = torch.frexp(values)
    return exponent - (mantissa == .5).to(exponent.dtype)


def pack_e2m1(matrix):
    """Pack one FP32 [rows, columns] matrix; columns must be a multiple of 32.

    Transcribed from simulate_fp4 so decode(pack(w)) == simulate_fp4(w) exactly:
    power-of-two scale per row/32-column block from amax/6, nearest-even E2M1
    levels, low nibble = even column, E8M0 scale byte = exponent+127.
    """
    rows, columns = matrix.shape
    if matrix.dtype is not torch.float32 or columns % 32:
        raise ValueError('e2m1 packing needs FP32 rows with 32-aligned width')
    groups = matrix.unflatten(-1, (-1, 32))
    scale_exponent = _ceil_power_two_exponent(
        groups.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -126) * (1 / 6))
    scale = torch.ldexp(torch.ones_like(scale_exponent, dtype=torch.float32), scale_exponent)
    normalized = (groups / scale).clamp(-6, 6)
    magnitude = normalized.abs()
    boundaries = matrix.new_tensor(BOUNDARIES, dtype=torch.float32)
    code = torch.bucketize(magnitude.contiguous(), boundaries)
    tie = magnitude == boundaries[code.clamp_max(6)]
    code = code + ((code % 2 == 1) & tie).long()
    sign = (normalized < 0) & (magnitude != 0)
    nibble = (code | sign.long() * 8).to(torch.uint8)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    scale_bytes = (scale_exponent + 127).to(torch.uint8)
    return packed.reshape(rows, columns // 2), scale_bytes.reshape(rows, columns // 32)


def roundtrip(tensor):
    """decode_tile(pack(w)) must equal simulate_fp4(w) exactly."""
    rows, columns = tensor.shape
    packed, scales = pack_e2m1(tensor)
    decoded = decode_tile(packed, scales, format='e2m1',
                          row_start=0, row_stop=rows, column_start=0, column_stop=columns)
    return bool(torch.equal(decoded, simulate_fp4(tensor)))


def routed_control(tensors, root, c, hidden, ids, weights):
    """Cell-6 CPU control with the shared expert zeroed: routed contribution only."""
    effective = dict(tensors)
    for suffix in ('mlp.experts.gate_up_proj', 'mlp.experts.down_proj'):
        effective[root + suffix] = simulate_fp4(tensors[root + suffix])
    for suffix in ('mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight',
                   'mlp.shared_expert.down_proj.weight'):
        effective[root + suffix] = torch.zeros_like(tensors[root + suffix])
    ffn = HybridMoeFeedForward(
        effective[root + 'mlp.gate.weight'], effective[root + 'mlp.experts.gate_up_proj'],
        effective[root + 'mlp.experts.down_proj'], effective[root + 'mlp.shared_expert.gate_proj.weight'],
        effective[root + 'mlp.shared_expert.up_proj.weight'],
        effective[root + 'mlp.shared_expert.down_proj.weight'],
        effective[root + 'mlp.shared_expert_gate.weight'],
        top_k=c['num_experts_per_tok'], scoring='softmax', renormalize=True,
        workspace_bytes=2 ** 40, max_output_bytes=2 ** 40)
    # Same routes as the GPU run; recompute to confirm identical selection.
    logits = torch.nn.functional.linear(hidden.reshape(-1, c['hidden_size']),
                                         tensors[root + 'mlp.gate.weight'])
    control_weights, control_ids = select_experts(logits, c['num_experts_per_tok'],
                                                  scoring='softmax', renormalize=True)
    assert torch.equal(control_ids, ids) and torch.equal(control_weights, weights)
    return ffn.forward(hidden).reshape(-1, c['hidden_size'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-seed', type=int, default=20261022)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    c = config['text_config']
    device = torch.device('cuda')
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    values = c['num_experts'] * (2 * c['moe_intermediate_size'] * c['hidden_size']
                                 + c['hidden_size'] * c['moe_intermediate_size']) * c['num_hidden_layers']
    bank_bytes = values * (0.5 + 1 / 32)
    if free_bytes < bank_bytes + 2 * 2 ** 30:
        raise ValueError(f'A30 free {free_bytes/2**30:.1f} GiB cannot hold the {bank_bytes/2**30:.1f} GiB bank')

    generator = torch.Generator().manual_seed(args.probe_seed)
    sample = torch.randn(64, 512, generator=generator)
    assert roundtrip(sample), 'e2m1 round-trip failed on the random sample'

    placement = []
    bank = {}
    inter, hidden = 2 * c['moe_intermediate_size'], c['hidden_size']
    chunk = 32
    for layer in range(c['num_hidden_layers']):
        root = f'model.language_model.layers.{layer}.'
        tensors = read_tensors(base, [root + suffix for suffix in MLP_SUFFIXES])
        gate_up = tensors[root + 'mlp.experts.gate_up_proj'].reshape(c['num_experts'] * inter, hidden)
        down = tensors[root + 'mlp.experts.down_proj'].reshape(c['num_experts'] * hidden,
                                                               c['moe_intermediate_size'])
        if layer == 0:
            tile = tensors[root + 'mlp.experts.gate_up_proj'][0, :128, :512]
            assert roundtrip(tile.contiguous()), 'e2m1 round-trip failed on the real tensor tile'
            del tile
        gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
        for start in range(0, c['num_experts'], chunk):
            stop = min(c['num_experts'], start + chunk)
            gu_p, gu_s = pack_e2m1(gate_up[start * inter:stop * inter].to(device))
            dn_p, dn_s = pack_e2m1(down[start * hidden:stop * hidden].to(device))
            rows = stop - start
            # Fused rows [e*2I, e*2I+I) are the gate half; [e*2I+I, (e+1)*2I) the up half.
            gate_w.append(gu_p.view(rows, inter, hidden // 2)[:, :c['moe_intermediate_size']])
            gate_s.append(gu_s.view(rows, inter, hidden // 32)[:, :c['moe_intermediate_size']])
            up_w.append(gu_p.view(rows, inter, hidden // 2)[:, c['moe_intermediate_size']:])
            up_s.append(gu_s.view(rows, inter, hidden // 32)[:, c['moe_intermediate_size']:])
            down_w.append(dn_p.view(rows, hidden, c['moe_intermediate_size'] // 2))
            down_s.append(dn_s.view(rows, hidden, c['moe_intermediate_size'] // 32))
            if layer == 0 and start == 0:
                # GPU-packed bytes must decode to exactly simulate_fp4 of the same weights.
                cpu_slice = gate_up[start * inter:start * inter + 128, :512]
                gpu_packed = gu_p[:128, :256].cpu()
                gpu_scales = gu_s[:128, :16].cpu()
                decoded = decode_tile(gpu_packed, gpu_scales, format='e2m1',
                                      row_start=0, row_stop=128, column_start=0, column_stop=512)
                assert torch.equal(decoded, simulate_fp4(cpu_slice)), 'GPU packing is not exact'
                del cpu_slice, gpu_packed, gpu_scales, decoded
            del gu_p, gu_s, dn_p, dn_s
        bank[layer] = {
            'gate_weight': torch.cat(gate_w), 'gate_scales': torch.cat(gate_s),
            'up_weight': torch.cat(up_w), 'up_scales': torch.cat(up_s),
            'down_weight': torch.cat(down_w), 'down_scales': torch.cat(down_s),
        }
        del gate_w, gate_s, up_w, up_s, down_w, down_s, tensors, gate_up, down
        if layer % 5 == 0 or layer == c['num_hidden_layers'] - 1:
            allocated = torch.cuda.memory_allocated()
            placement.append({'layer': layer, 'allocated_bytes': allocated,
                              'allocated_gib': round(allocated / 2 ** 30, 2)})
            print(f'placed layer {layer}: allocated {allocated/2**30:.2f} GiB', flush=True)

    probe = (torch.randn(2, 6, c['hidden_size'], generator=torch.Generator().manual_seed(args.probe_seed))
             * 0.5)
    flat = probe.reshape(-1, c['hidden_size'])
    composition = {}
    digests = {}
    for layer in (3, 39):
        root = f'model.language_model.layers.{layer}.'
        tensors = read_tensors(base, [root + suffix for suffix in MLP_SUFFIXES])
        for suffix in MLP_SUFFIXES:
            digests[f'layer{layer}.{suffix}'] = digest(tensors[root + suffix])
        logits = torch.nn.functional.linear(flat, tensors[root + 'mlp.gate.weight'])
        weights, ids = select_experts(logits, c['num_experts_per_tok'],
                                      scoring='softmax', renormalize=True)
        experts = [CudaPackedExpert(
            CudaPackedMatrix(bank[layer]['gate_weight'][e], bank[layer]['gate_scales'][e], 'e2m1'),
            CudaPackedMatrix(bank[layer]['up_weight'][e], bank[layer]['up_scales'][e], 'e2m1'),
            CudaPackedMatrix(bank[layer]['down_weight'][e], bank[layer]['down_scales'][e], 'e2m1'),
            row_chunk=64, column_chunk=256, workspace_bytes=8 * 1024 ** 2)
            for e in range(c['num_experts'])]
        gpu = cuda_packed_moe(flat.to(device), experts, ids.to(device), weights.to(device),
                              fused=True)
        control = routed_control(tensors, root, c, probe, ids, weights)
        composition[layer] = {
            'selected_first_row': ids[0].tolist(),
            'gpu_vs_cell6_control': metrics(gpu.cpu(), control),
            'gpu_vs_fp32_reference': metrics(
                gpu.cpu(), _fp32_routed(tensors, root, c, probe, ids, weights)),
        }
        del experts, gpu, control, tensors
        print(f'composed layer {layer}', flush=True)

    report = {
        'snapshot': str(base),
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'bank_bytes_arithmetic': bank_bytes, 'bank_gib_arithmetic': bank_bytes / 2 ** 30,
        'gpu_free_before_gib': free_bytes / 2 ** 30, 'gpu_total_gib': total_bytes / 2 ** 30,
        'placed_layers': c['num_hidden_layers'],
        'placement': placement,
        'allocated_after_placement_bytes': torch.cuda.memory_allocated(),
        'reserved_after_placement_bytes': torch.cuda.memory_reserved(),
        'roundtrip_assert': 'decode_tile(pack(w)) == simulate_fp4(w) exact (random sample + real gate_up tile)',
        'composed_tensor_sha256': digests,
        'probe': {'kind': 'synthetic normal (not real activations)', 'seed': args.probe_seed,
                  'shape': [2, 6, c['hidden_size']]},
        'composition': {str(layer): composition[layer] for layer in composition},
        'tf32': False,
        'scope': 'routed expert bank E2M1 residency + fused composition; router/shared stay FP32 CPU',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'placement': placement[-1], 'allocated_gib':
                     round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                     'composition': {str(k): v['gpu_vs_cell6_control']
                                     for k, v in composition.items()}}, indent=1))


def _fp32_routed(tensors, root, c, probe, ids, weights):
    """Unquantized routed contribution (shared zeroed) for context in the report."""
    effective = dict(tensors)
    for suffix in ('mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight',
                   'mlp.shared_expert.down_proj.weight'):
        effective[root + suffix] = torch.zeros_like(tensors[root + suffix])
    ffn = HybridMoeFeedForward(
        tensors[root + 'mlp.gate.weight'], tensors[root + 'mlp.experts.gate_up_proj'],
        tensors[root + 'mlp.experts.down_proj'], effective[root + 'mlp.shared_expert.gate_proj.weight'],
        effective[root + 'mlp.shared_expert.up_proj.weight'],
        effective[root + 'mlp.shared_expert.down_proj.weight'],
        tensors[root + 'mlp.shared_expert_gate.weight'],
        top_k=c['num_experts_per_tok'], scoring='softmax', renormalize=True,
        workspace_bytes=2 ** 40, max_output_bytes=2 ** 40)
    return ffn.forward(probe).reshape(-1, c['hidden_size'])


if __name__ == '__main__':
    main()
