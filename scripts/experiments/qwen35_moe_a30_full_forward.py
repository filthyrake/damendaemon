"""Single-GPU full-model text forward for qwen3_5_moe (#14, cell 8).

Runs all 40 text layers of the real Qwen3.6-35B-A3B snapshot in one guarded
A30 process with the cell-7 E2M1 expert bank resident and the non-expert
weights (norms, gated-deltanet linear attention, full attention, routers,
shared experts, combine gates) resident in FP32 (~5.8 GiB arithmetic; the
embedding stays host-side - only the 32-token probe needs it). The forward
is the cell-4/5 ported math on CUDA, with per-layer MoE composition through
the prepared CUDA fused path and FP32 routing (cell-3 pinned policy).

Compared against the cell-5 host 40-layer run re-executed with hidden dumps
at the same probe seed (20261023): the primary number is GPU vs host-ours
final hidden - the real-weight E2M1 quantization error compounded through
all 40 residual streams, which cell 6 could only bound per-layer. Checkpoint
layers 0/3/9/20/39 record the compounding curve; the upstream-eager
comparison is included for context.

Expected class: per-layer FFN deviations ~2e-03 accumulating through 40
layers - NOT the 3e-05 FP32 baseline. Metrics recorded exactly as measured;
no quality/throughput claim. Run under guard_shared_cuda with the A30 UUID
without -O (asserts exactness of the probe reproduction).
"""
if not __debug__:
    raise RuntimeError('qwen35_moe_a30_full_forward asserts probe reproduction; run without -O')

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
from damendaemon.cuda_packed_moe import cuda_packed_moe
from damendaemon.expert_selection import select_experts
from scripts.experiments.qwen35_moe_a30_residency import pack_e2m1
from scripts.experiments.qwen35_moe_text_executor import (
    digest, layer_names, metrics, read_tensors)

CHECKPOINTS = (0, 3, 9, 20, 39)


def rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gpu_linear_attn(w, x, c):
    b, t, _ = x.shape
    lk, lv, dk, dv = c['linear_num_key_heads'], c['linear_num_value_heads'], \
        c['linear_key_head_dim'], c['linear_value_head_dim']
    width, kernel = 2 * lk * dk + lv * dv, c['linear_conv_kernel_dim']
    qkv = F.linear(x, w['qkv']).transpose(1, 2)
    convolved = F.conv1d(qkv, w['conv'], padding=kernel - 1, groups=width)[..., :t]
    qkv = F.silu(convolved).transpose(1, 2)
    q, k, v = qkv.split((lk * dk, lk * dk, lv * dv), -1)
    q, k = [a.reshape(b, t, lk, dk) for a in (q, k)]
    v = v.reshape(b, t, lv, dv)
    beta = F.linear(x, w['b']).sigmoid()
    decay = -w['A_log'].exp() * F.softplus(F.linear(x, w['a']) + w['dt'])
    z = F.linear(x, w['z']).reshape(b, t, lv, dv)
    q, k = [l2norm(a) for a in (q, k)]
    q, k = [a.repeat_interleave(lv // lk, dim=2) for a in (q, k)]
    q = q * (dk ** -0.5)
    state = torch.zeros(b, lv, dk, dv, dtype=torch.float32, device=x.device)
    outputs = []
    for step in range(t):
        state = state * decay[:, step][..., None, None].exp()
        delta = (v[:, step] - (state * k[:, step, :, :, None]).sum(-2)) * beta[:, step, :, None]
        state = state + k[:, step, :, :, None] * delta.unsqueeze(-2)
        outputs.append((state * q[:, step, :, :, None]).sum(-2))
    out = torch.stack(outputs, 1)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + c['rms_norm_eps']) * w['gated']
    out = out * F.silu(z)
    return F.linear(out.reshape(b, t, lv * dv), w['out'])


def gpu_full_attn(w, x, cos, sin, c):
    b, t, _ = x.shape
    heads, kh, dim, rd = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim'], c['rotary_dim']
    eps = c['rms_norm_eps']
    q = F.linear(x, w['q']).view(b, t, heads, 2 * dim)
    q, gate = q.chunk(2, -1)
    gate = gate.reshape(b, t, heads * dim)
    k = F.linear(x, w['k']).view(b, t, kh, dim)
    v = F.linear(x, w['v']).view(b, t, kh, dim)
    q = rms(q, w['q_norm'], eps)
    k = rms(k, w['k_norm'], eps)

    def rotate(a):
        rotated, rest = a[..., :rd], a[..., rd:]
        left, right = rotated.chunk(2, -1)
        return torch.cat((rotated * cos + torch.cat((-right, left), -1) * sin, rest), -1)

    q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
    v = v.transpose(1, 2).repeat_interleave(heads // kh, 1)
    scores = torch.matmul(q, k.repeat_interleave(heads // kh, 1).transpose(2, 3)) * (dim ** -0.5)
    causal = torch.full((t, t), float('-inf'), device=x.device).triu(1)
    weights = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
    out = torch.matmul(weights, v).transpose(1, 2).reshape(b, t, heads * dim)
    return F.linear(out * gate.sigmoid(), w['o'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--host-dumps', required=True,
                        help='directory with the host run layer/final .pt dumps')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default=None,
                        help='optional path to cache the packed bank (16 GiB) for fast reruns')
    parser.add_argument('--layer-start', type=int, default=0)
    parser.add_argument('--layer-end', type=int, default=None, help='exclusive; default all')
    parser.add_argument('--hidden-in', default=None, help='load a carried hidden state [1,32,hidden]')
    parser.add_argument('--hidden-out', default=None, help='save the hidden state after the last layer')
    parser.add_argument('--fused-route-chunk', type=int, default=256,
                        help='routes per fused composition chunk (256 removes per-chunk tile re-decode)')
    parser.add_argument('--probe-seed', type=int, default=20261023)
    args = parser.parse_args()
    if args.layer_end is None:
        args.layer_end = 10 ** 9
    if not 0 <= args.layer_start < args.layer_end:
        raise ValueError('invalid layer range')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    c = config['text_config']
    c['rotary_dim'] = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])
    layer_end = min(args.layer_end, c['num_hidden_layers'])
    device = torch.device('cuda')

    free_bytes, _ = torch.cuda.mem_get_info()
    expert_values = c['num_experts'] * (2 * c['moe_intermediate_size'] * c['hidden_size']
                                        + c['hidden_size'] * c['moe_intermediate_size'])
    bank_bytes = expert_values * (0.5 + 1 / 32) * c['num_hidden_layers']
    nonexpert_bytes = 4 * (  # FP32: attention + norms + routers + shared experts + gates
        c['num_hidden_layers'] * (c['num_experts'] * c['hidden_size']
                                  + 3 * c['shared_expert_intermediate_size'] * c['hidden_size']
                                  + 2 * c['hidden_size'])
        + 30 * ((2 * c['linear_num_key_heads'] * c['linear_key_head_dim']
                 + c['linear_num_value_heads'] * c['linear_value_head_dim'] * 2) * c['hidden_size'])
        + 10 * ((2 * c['num_attention_heads'] + 2 * c['num_key_value_heads'])
                * c['head_dim'] * c['hidden_size']))
    if free_bytes < bank_bytes + nonexpert_bytes + 2 ** 30:
        raise ValueError(f'A30 free {free_bytes/2**30:.1f} GiB cannot hold bank '
                         f'{bank_bytes/2**30:.1f} + non-expert {nonexpert_bytes/2**30:.1f} GiB')

    inter, hidden = 2 * c['moe_intermediate_size'], c['hidden_size']
    chunk = 32
    placement = []
    bank = {}
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        bank = torch.load(cache_path, map_location=device, weights_only=True)
        allocated = torch.cuda.memory_allocated()
        placement.append({'loaded_from_cache': str(cache_path), 'allocated_gib': round(allocated / 2 ** 30, 2)})
        print(f'bank loaded from cache: {allocated/2**30:.2f} GiB', flush=True)
    else:
        for layer in range(c['num_hidden_layers']):
            kind = c['layer_types'][layer]
            root = f'model.language_model.layers.{layer}.'
            tensors = read_tensors(base, layer_names(layer, kind))
            gate_up = tensors[root + 'mlp.experts.gate_up_proj'].reshape(c['num_experts'] * inter, hidden)
            down = tensors[root + 'mlp.experts.down_proj'].reshape(c['num_experts'] * hidden,
                                                                   c['moe_intermediate_size'])
            gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
            for start in range(0, c['num_experts'], chunk):
                stop = min(c['num_experts'], start + chunk)
                gu_p, gu_s = pack_e2m1(gate_up[start * inter:stop * inter].to(device))
                dn_p, dn_s = pack_e2m1(down[start * hidden:stop * hidden].to(device))
                rows = stop - start
                gate_w.append(gu_p.view(rows, inter, hidden // 2)[:, :c['moe_intermediate_size']])
                gate_s.append(gu_s.view(rows, inter, hidden // 32)[:, :c['moe_intermediate_size']])
                up_w.append(gu_p.view(rows, inter, hidden // 2)[:, c['moe_intermediate_size']:])
                up_s.append(gu_s.view(rows, inter, hidden // 32)[:, c['moe_intermediate_size']:])
                down_w.append(dn_p.view(rows, hidden, c['moe_intermediate_size'] // 2))
                down_s.append(dn_s.view(rows, hidden, c['moe_intermediate_size'] // 32))
                del gu_p, gu_s, dn_p, dn_s
            bank[layer] = (torch.cat(gate_w), torch.cat(gate_s), torch.cat(up_w), torch.cat(up_s),
                           torch.cat(down_w), torch.cat(down_s))
            del tensors, gate_up, down, gate_w, gate_s, up_w, up_s, down_w, down_s
            if layer % 5 == 0 or layer == c['num_hidden_layers'] - 1:
                allocated = torch.cuda.memory_allocated()
                placement.append({'layer': layer, 'allocated_gib': round(allocated / 2 ** 30, 2)})
                print(f'placed layer {layer}: {allocated/2**30:.2f} GiB', flush=True)
        if cache_path is not None:
            torch.save(bank, cache_path)
            print(f'bank cached to {cache_path}', flush=True)

    # Final norm on GPU; embedding host-side (probe only needs 32 rows).
    final_norm_weight = read_tensors(
        base, ['model.language_model.norm.weight'])['model.language_model.norm.weight'].to(device)
    embed = read_tensors(base, ['model.language_model.embed_tokens.weight'])['model.language_model.embed_tokens.weight']
    generator = torch.Generator().manual_seed(args.probe_seed)
    ids = torch.randint(0, c['vocab_size'], (1, 32), generator=generator)
    if args.hidden_in:
        hidden_states = torch.load(args.hidden_in, map_location=device, weights_only=True)
    else:
        hidden_states = embed[ids].to(device)
    del embed

    theta = c['rope_parameters']['rope_theta']
    rd = c['rotary_dim']
    inv = theta ** (-torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
    freqs = torch.outer(torch.arange(32, device=device, dtype=torch.float32), inv)
    angles = torch.cat((freqs, freqs), -1).view(1, 32, rd).unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    eps = c['rms_norm_eps']
    checkpoints = {}
    peak_forward_allocated = 0
    with torch.inference_mode():
        for layer in range(args.layer_start, layer_end):
            kind = c['layer_types'][layer]
            root = f'model.language_model.layers.{layer}.'
            # Non-expert FP32 weights stream per layer (used exactly once);
            # full residency (~21.7 GiB bank+weights) exceeds the guard's
            # 21.5 GiB envelope on the 24-GiB A30 - recorded, not bypassed.
            tensors = read_tensors(base, layer_names(layer, kind))
            gate_w, gate_s, up_w, up_s, down_w, down_s = bank[layer]
            experts = [CudaPackedExpert(
                CudaPackedMatrix(gate_w[e], gate_s[e], 'e2m1'),
                CudaPackedMatrix(up_w[e], up_s[e], 'e2m1'),
                CudaPackedMatrix(down_w[e], down_s[e], 'e2m1'),
                row_chunk=64, column_chunk=256, workspace_bytes=8 * 1024 ** 2)
                for e in range(c['num_experts'])]
            e = {
                'experts': experts,
                'router': tensors[root + 'mlp.gate.weight'].to(device),
                'sg': tensors[root + 'mlp.shared_expert.gate_proj.weight'].to(device),
                'su': tensors[root + 'mlp.shared_expert.up_proj.weight'].to(device),
                'sd': tensors[root + 'mlp.shared_expert.down_proj.weight'].to(device),
                'seg': tensors[root + 'mlp.shared_expert_gate.weight'].to(device),
                'ln1': tensors[root + 'input_layernorm.weight'].to(device),
                'ln2': tensors[root + 'post_attention_layernorm.weight'].to(device),
            }
            if kind == 'linear_attention':
                e.update({
                    'qkv': tensors[root + 'linear_attn.in_proj_qkv.weight'].to(device),
                    'z': tensors[root + 'linear_attn.in_proj_z.weight'].to(device),
                    'b': tensors[root + 'linear_attn.in_proj_b.weight'].to(device),
                    'a': tensors[root + 'linear_attn.in_proj_a.weight'].to(device),
                    'conv': tensors[root + 'linear_attn.conv1d.weight'].to(device),
                    'dt': tensors[root + 'linear_attn.dt_bias'].to(device),
                    'A_log': tensors[root + 'linear_attn.A_log'].to(device),
                    'gated': tensors[root + 'linear_attn.norm.weight'].to(device),
                    'out': tensors[root + 'linear_attn.out_proj.weight'].to(device),
                })
            else:
                e.update({
                    'q': tensors[root + 'self_attn.q_proj.weight'].to(device),
                    'k': tensors[root + 'self_attn.k_proj.weight'].to(device),
                    'v': tensors[root + 'self_attn.v_proj.weight'].to(device),
                    'o': tensors[root + 'self_attn.o_proj.weight'].to(device),
                    'q_norm': tensors[root + 'self_attn.q_norm.weight'].to(device),
                    'k_norm': tensors[root + 'self_attn.k_norm.weight'].to(device),
                })
            del tensors
            normed = rms(hidden_states, e['ln1'], eps)
            mixer = (gpu_full_attn(e, normed, cos, sin, c) if kind == 'full_attention'
                     else gpu_linear_attn(e, normed, c))
            hidden_states = hidden_states + mixer
            normed = rms(hidden_states, e['ln2'], eps)
            flat = normed.reshape(-1, hidden)
            logits = F.linear(flat, e['router'])
            weights, selected = select_experts(logits, c['num_experts_per_tok'],
                                               scoring='softmax', renormalize=True)
            routed = cuda_packed_moe(flat, e['experts'], selected, weights, fused=True,
                                    fused_route_chunk=args.fused_route_chunk)
            shared = F.linear(F.silu(F.linear(flat, e['sg'])) * F.linear(flat, e['su']), e['sd'])
            combine = torch.sigmoid(F.linear(flat, e['seg']))
            hidden_states = hidden_states + (routed + shared * combine).reshape(1, 32, hidden)
            if layer in CHECKPOINTS:
                checkpoints[layer] = hidden_states.detach().clone()
            peak_forward_allocated = max(peak_forward_allocated, torch.cuda.memory_allocated())
            del e
            print(f'forward layer {layer} done', flush=True)
        if args.hidden_out:
            torch.save(hidden_states.detach().clone(), args.hidden_out)
            print(f'hidden state saved to {args.hidden_out}', flush=True)

    dumps = Path(args.host_dumps)
    # weights_only: these files contain only tensor dictionaries written by the
    # host dump run; no pickled objects are expected or accepted.
    assert torch.equal(ids, torch.load(dumps / 'probe.pt', map_location='cpu', weights_only=True)), \
        'probe mismatch'
    curve = {}
    for layer in sorted(checkpoints):
        host = torch.load(dumps / f'layer{layer}.pt', map_location='cpu', weights_only=True)['ours']
        curve[layer] = metrics(checkpoints[layer].cpu(), host)
    report = {
        'snapshot': str(base),
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'layer_range': [args.layer_start, layer_end],
        'placement': placement,
        'allocated_after_placement_gib': round(torch.cuda.memory_allocated() / 2 ** 30, 2),
        'reserved_after_placement_gib': round(torch.cuda.memory_reserved() / 2 ** 30, 2),
        'bank_bytes_arithmetic': bank_bytes,
        'nonexpert_bytes_arithmetic': nonexpert_bytes,
        'nonexpert_residency': 'streamed per layer (~150 MiB, used once); full FP32 residency '
                               'of bank+weights (~21.7 GiB) exceeds the guard envelope (24 - 2.5 reserve)',
        'peak_forward_allocated_gib': round(peak_forward_allocated / 2 ** 30, 2),
        'probe': {'kind': 'synthetic random token IDs (not real text)', 'seed': args.probe_seed,
                  'shape': [1, 32, c['hidden_size']]},
        'gpu_vs_host_ours_checkpoints': {str(k): curve[k] for k in curve},
        'tf32': False,
        'scope': 'single-GPU forward: E2M1 expert bank + FP32 non-expert weights streamed per layer; '
                 'embedding host-side; CPU control is the cell-5 host run',
    }
    if layer_end == c['num_hidden_layers']:
        host_final = torch.load(dumps / 'final.pt', map_location='cpu', weights_only=True)
        gpu_final = rms(hidden_states, final_norm_weight, eps)
        report['gpu_vs_host_ours_final'] = metrics(gpu_final.cpu(), host_final['final_ours'])
        report['gpu_vs_host_upstream_final'] = metrics(gpu_final.cpu(), host_final['final_theirs'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'layer_range': report['layer_range'],
                      'final_gpu_vs_host_ours': report.get('gpu_vs_host_ours_final'),
                      'final_gpu_vs_host_upstream': report.get('gpu_vs_host_upstream_final'),
                      'checkpoints': report['gpu_vs_host_ours_checkpoints'],
                      'allocated_gib': report['allocated_after_placement_gib']}, indent=1))


if __name__ == '__main__':
    main()
