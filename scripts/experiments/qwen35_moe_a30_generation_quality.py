"""Generation-quality gate for the E4M3 expert bank on real text (#14, cell 10).

Cell 9 established that hidden-state fidelity is structural to weights-only
expert quantization: deviations decorrelate to O(1) via route flips. The
arbiter for deployability is generation quality on REAL text, measured here
against the host FP32 reference (the cell-5 ported math, verified against
the eager upstream stack at rms_rel 2.4e-04 on the same snapshot).

Prompts: four fixed English excerpts from the Qwen3.6-35B-A3B model card
(real natural text, deterministic, no network). Tokenized with the snapshot
tokenizer; truncated to 64 tokens each; batch [4, 64].

PRE-DECLARED ACCEPTANCE THRESHOLDS (no post-hoc relaxation; failure is
recorded as failure):
  1. greedy top-1 next-token agreement (positions >= 8, all prompts,
     quantized vs reference): >= 0.80
  2. mean per-position KL(reference || quantized) over the full vocabulary:
     <= 1.0 nat
  3. perplexity ratio (quantized / reference, teacher-forced on the same
     prompt tokens): <= 1.10

Modes (run reference first, then candidate under guard_shared_cuda on the
A30 UUID without -O):
  reference: host FP32 forward (streamed per layer) + lm_head logits on
             CPU; saves {ids, logits} for the candidate run.
  candidate: GPU E4M3 per-layer streamed packing forward (bank never
             resident, cell-9 structure) + lm_head on GPU; loads the
             reference, computes the three metrics, writes the report with
             pass/fail recorded exactly as measured.

Claim scope: these three metrics on these four prompts for this snapshot.
Not claimed: throughput, longer contexts, other prompt sets, or any
threshold not listed above.
"""
if not __debug__:
    raise RuntimeError('qwen35_moe_a30_generation_quality asserts probe equality; run without -O')

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward
from scripts.experiments.qwen35_moe_a30_full_forward import (
    pack_layer_experts, gpu_linear_attn, gpu_full_attn, rms)
from scripts.experiments.qwen35_moe_text_executor import (
    layer_names, ours_full, ours_linear, read_tensors)

PROMPTS = (
    'Qwen3.6 is a hybrid multimodal model built on a hybrid architecture that '
    'combines linear attention for long-context efficiency with full attention '
    'for precise recall. The model ships in several sizes and supports a very '
    'long context length, which makes it suitable for document understanding.',
    'The harbor terminal benchmark ran with a three hour timeout on a machine '
    'with thirty two CPU cores and forty eight gigabytes of memory. Sampling '
    'used temperature one point zero, top p ninety five percent and top k '
    'twenty. The reported score is the average of five independent runs.',
    'A mixture of experts layer routes each token to a small number of expert '
    'networks using a learned router. Because only a few experts are active '
    'per token, the total parameter count can grow without a matching increase '
    'in computation cost per token.',
    'Please summarize the following situation in one sentence. The database '
    'server restarted twice last night, first at midnight and again at four '
    'in the morning, and each time the cache warmed up slowly which delayed '
    'the morning reports for the sales team.',
)
MAX_TOKENS = 64
WARMUP_POSITIONS = 8
GATES = {'greedy_agreement_min': 0.80, 'kl_max': 1.0, 'perplexity_ratio_max': 1.10}
MLP_SUFFIXES = ('mlp.gate.weight', 'mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
                'mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight',
                'mlp.shared_expert.down_proj.weight', 'mlp.shared_expert_gate.weight')


def tokenize(snapshot):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    rows = tokenizer(list(PROMPTS), add_special_tokens=True)['input_ids']
    rows = [row[:MAX_TOKENS] for row in rows]
    length = min(len(row) for row in rows)
    if length < 32:
        raise ValueError('prompts must tokenize to at least 32 tokens each')
    # Trim to the shortest row: the ported reference math assumes uniform
    # causal lengths with no padding.
    ids = torch.tensor([row[:length] for row in rows], dtype=torch.int64)
    return ids


def load_lm_head(base):
    head = read_tensors(base, ['lm_head.weight'])['lm_head.weight']
    if head.shape[0] < 240000:
        raise ValueError('unexpected lm_head geometry')
    return head


def reference_forward(base, c, ids):
    """Host FP32 forward with the cell-5 ported math; per-layer streaming."""
    c = dict(c)
    c['rotary_dim'] = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])
    embed = read_tensors(base, ['model.language_model.embed_tokens.weight'])['model.language_model.embed_tokens.weight']
    hidden = embed[ids]
    del embed
    rd = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])
    theta = c['rope_parameters']['rope_theta']
    inv = theta ** (-torch.arange(0, rd, 2, dtype=torch.float32) / rd)
    freqs = torch.outer(torch.arange(ids.shape[1], dtype=torch.float32), inv)
    angles = torch.cat((freqs, freqs), -1).view(1, ids.shape[1], rd).unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    eps = c['rms_norm_eps']
    with torch.inference_mode():
        for layer in range(c['num_hidden_layers']):
            kind = c['layer_types'][layer]
            root = f'model.language_model.layers.{layer}.'
            tensors = read_tensors(base, layer_names(layer, kind))
            ffn = HybridMoeFeedForward(
                tensors[root + 'mlp.gate.weight'], tensors[root + 'mlp.experts.gate_up_proj'],
                tensors[root + 'mlp.experts.down_proj'], tensors[root + 'mlp.shared_expert.gate_proj.weight'],
                tensors[root + 'mlp.shared_expert.up_proj.weight'],
                tensors[root + 'mlp.shared_expert.down_proj.weight'],
                tensors[root + 'mlp.shared_expert_gate.weight'],
                top_k=c['num_experts_per_tok'], scoring='softmax', renormalize=True,
                workspace_bytes=2 ** 40, max_output_bytes=2 ** 40)
            normed = rms(hidden, tensors[root + 'input_layernorm.weight'], eps)
            mixer = (ours_full(tensors, root, normed, cos, sin, c) if kind == 'full_attention'
                     else ours_linear(tensors, root, normed, c))
            hidden = hidden + mixer
            hidden = hidden + ffn.forward(rms(hidden, tensors[root + 'post_attention_layernorm.weight'], eps))
            del tensors, ffn
        final = rms(hidden, read_tensors(base, ['model.language_model.norm.weight'])
                    ['model.language_model.norm.weight'], eps)
    head = load_lm_head(base)
    logits = F.linear(final.reshape(-1, c['hidden_size']), head)
    return logits


def candidate_forward(base, c, ids, device, *, route_chunk, max_gather_bytes):
    """GPU E4M3 per-layer streamed forward (cell-9 structure) + lm_head on GPU."""
    from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
    from damendaemon.cuda_packed_moe import cuda_packed_moe
    from damendaemon.expert_selection import select_experts
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = dict(c)
    c['rotary_dim'] = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])
    embed = read_tensors(base, ['model.language_model.embed_tokens.weight'])['model.language_model.embed_tokens.weight']
    hidden = embed[ids].to(device)
    del embed
    tokens = ids.shape[1]
    rd = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])
    theta = c['rope_parameters']['rope_theta']
    inv = theta ** (-torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
    freqs = torch.outer(torch.arange(tokens, device=device, dtype=torch.float32), inv)
    angles = torch.cat((freqs, freqs), -1).view(1, tokens, rd).unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    eps = c['rms_norm_eps']
    hidden_dim = c['hidden_size']
    peak = 0
    with torch.inference_mode():
        for layer in range(c['num_hidden_layers']):
            kind = c['layer_types'][layer]
            root = f'model.language_model.layers.{layer}.'
            tensors = read_tensors(base, layer_names(layer, kind))
            gate_w, gate_s, up_w, up_s, down_w, down_s = pack_layer_experts(
                tensors, root, c, device, 'e4m3fn', verify=layer == 0)
            experts = [CudaPackedExpert(
                CudaPackedMatrix(gate_w[e], gate_s[e], 'e4m3fn'),
                CudaPackedMatrix(up_w[e], up_s[e], 'e4m3fn'),
                CudaPackedMatrix(down_w[e], down_s[e], 'e4m3fn'),
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
            normed = rms(hidden, e['ln1'], eps)
            mixer = (gpu_full_attn(e, normed, cos, sin, c) if kind == 'full_attention'
                     else gpu_linear_attn(e, normed, c))
            hidden = hidden + mixer
            flat = rms(hidden, e['ln2'], eps).reshape(-1, hidden_dim)
            logits = F.linear(flat, e['router'])
            weights, selected = select_experts(logits, c['num_experts_per_tok'],
                                                scoring='softmax', renormalize=True)
            routed = cuda_packed_moe(flat, e['experts'], selected, weights, fused=True,
                                     fused_route_chunk=route_chunk,
                                     max_gather_bytes=max_gather_bytes)
            shared = F.linear(F.silu(F.linear(flat, e['sg'])) * F.linear(flat, e['su']), e['sd'])
            combine = torch.sigmoid(F.linear(flat, e['seg']))
            hidden = hidden + (routed + shared * combine).reshape(ids.shape[0], tokens, hidden_dim)
            peak = max(peak, torch.cuda.memory_allocated())
            del e, gate_w, gate_s, up_w, up_s, down_w, down_s
            if layer % 5 == 0 or layer == c['num_hidden_layers'] - 1:
                print(f'candidate layer {layer}: peak {peak/2**30:.2f} GiB', flush=True)
        norm_w = read_tensors(base, ['model.language_model.norm.weight'])['model.language_model.norm.weight'].to(device)
        final = rms(hidden, norm_w, eps)
        head = load_lm_head(base).to(device)
        out_logits = F.linear(final.reshape(-1, hidden_dim), head).cpu()
    return out_logits, round(peak / 2 ** 30, 2)


def evaluate(ref_logits, cand_logits, ids):
    """Pre-declared metrics; agreement/KL/NLL all over positions [warmup, last-1)."""
    rows, positions = ids.shape[0], ids.shape[1]
    ref = ref_logits.view(rows, positions, -1).float()
    cand = cand_logits.view(rows, positions, -1).float()
    ref_logp = F.log_softmax(ref, -1)
    cand_logp = F.log_softmax(cand, -1)
    agree = (ref.argmax(-1) == cand.argmax(-1))[:, WARMUP_POSITIONS:positions - 1]
    agreement = float(agree.float().mean())
    kl = (ref_logp.exp() * (ref_logp - cand_logp)).sum(-1)
    mean_kl = float(kl[:, WARMUP_POSITIONS:positions - 1].mean())
    nll = []
    for source in (ref_logp, cand_logp):
        target = ids[:, WARMUP_POSITIONS + 1:positions]
        picked = source[:, WARMUP_POSITIONS:positions - 1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll.append(float(-picked.mean()))
    ratio = float(torch.exp(torch.tensor(nll[1] - nll[0])))
    return {'greedy_agreement': agreement, 'mean_kl_nats': mean_kl,
            'perplexity_ratio': ratio,
            'reference_nll': nll[0], 'candidate_nll': nll[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--mode', choices=('reference', 'candidate'), required=True)
    parser.add_argument('--fused-route-chunk', type=int, default=1536,
                        help='routes per fused composition chunk; 1536 = one chunk per layer')
    parser.add_argument('--max-gather-bytes', type=int, default=256 * 1024 * 1024,
                        help='fused gathered-tile workspace cap (route chunk x 64x256 tiles)')
    parser.add_argument('--reference', default='/tmp/dd14-a30/quality-reference.pt',
                        help='reference .pt (written in reference mode, read in candidate mode)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    c = config['text_config']
    ids = tokenize(base)
    ids_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()

    if args.mode == 'reference':
        logits = reference_forward(base, c, ids)
        Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'ids': ids, 'logits': logits}, args.reference)
        print(json.dumps({'mode': 'reference', 'prompt_rows': list(ids.shape),
                          'ids_sha256': ids_sha}, indent=1))
        return

    saved = torch.load(args.reference, map_location='cpu', weights_only=True)
    assert torch.equal(ids, saved['ids']), 'tokenized prompts differ between runs'
    cand_logits, peak_gib = candidate_forward(base, c, ids, torch.device('cuda'),
                                              route_chunk=args.fused_route_chunk,
                                              max_gather_bytes=args.max_gather_bytes)
    result = evaluate(saved['logits'], cand_logits, ids)
    gates = {
        'greedy_agreement': result['greedy_agreement'] >= GATES['greedy_agreement_min'],
        'mean_kl_nats': result['mean_kl_nats'] <= GATES['kl_max'],
        'perplexity_ratio': result['perplexity_ratio'] <= GATES['perplexity_ratio_max'],
    }
    report = {
        'snapshot': str(base),
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'prompts': {'kind': 'fixed excerpts of the Qwen3.6-35B-A3B model card (real text)',
                    'count': len(PROMPTS), 'max_tokens': MAX_TOKENS,
                    'warmup_positions': WARMUP_POSITIONS, 'ids_sha256': ids_sha},
        'predeclared_gates': GATES,
        'metrics': result,
        'gate_results': gates,
        'expert_format': 'e4m3fn',
        'expert_residency': 'single-layer streamed packing (bank never resident)',
        'fused_route_chunk': args.fused_route_chunk,
        'max_gather_bytes': args.max_gather_bytes,
        'peak_gpu_gib': peak_gib,
        'tf32': False,
        'reference': 'host FP32 ported math (cell 5; verified vs eager upstream at rms_rel 2.4e-04)',
        'scope': 'three pre-declared metrics on four fixed model-card prompts for this snapshot; '
                 'no throughput, long-context, or other-prompt-set claims',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'metrics': result, 'gate_results': gates, 'peak_gpu_gib': peak_gib}, indent=1))


if __name__ == '__main__':
    main()
