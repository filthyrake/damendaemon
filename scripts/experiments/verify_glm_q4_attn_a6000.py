"""Bounded A6000 cell: GLM-5.3 Q4_0 attention projections vs shipped-FP8 oracle (#719 rung 4b).

Loads one real sparse-attention layer (default 6) of the official GLM-5.3
checkpoint from the staged NAS original and runs the PRODUCTION CUDA
flash-sparse attention composition (CudaFlashSparseAttention + CudaFlashIndexer
from cuda_glm_flash_sparse / cuda_glm_flash_indexer) TWICE:

- oracle: the four packed attention projections (q_a_proj, q_b_proj,
  kv_a_proj_with_mqa, o_proj) placed directly as shipped blockwise-E4M3-FP8
  (CudaPackedProjection format 'e4m3fn_f32' — byte-identical to the
  checkpoint's uint8 codes + FP32 128x128 scale_inv grids)
- candidate: the same weights packed to canonical Q4_0 on device
  (damendaemon.glm_q4 decode + quantization.quantize_q4), admitted by the
  widened GLM attention packed-projection admission (uniform 'q4_0_f16')

Everything else is IDENTICAL in both runs — kv_b_proj, both layernorms and
the whole indexer ship dense floating and are shared by reference, so the
token selection must be bit-identical (selection-exactness #668 is
structural: the indexer never goes through packed projections).

Pre-declared gates (on #719 before launch; drift envelope derives from rung
1's measured weight-level q4-of-FP8 drift on attention tensors, 9.55-10.24%
rel RMS — the standing gate is the 1.5x family envelope; no token or quality
claim, no throughput claim):

1. ATTENTION PARITY: full flash-sparse layer output (projection, indexer
   selection, latent attention, output projection) q4 vs fp8 rel RMS <= 0.15
   and max_abs <= 0.15 x oracle absmax.
2. SELECTION EXACTNESS: the selected-token index sets must be bit-identical
   between the two runs (identical dense-FP32 indexer, identical inputs).
3. FINITENESS: both outputs finite (also enforced inside the layer).
4. MEMORY: q4 projection bytes < fp8 projection bytes; peak reserved recorded.
"""
import argparse
import json
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=64)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    from safetensors import safe_open
    from damendaemon.cuda_glm_flash_sparse import CudaFlashSparseAttention
    from damendaemon.cuda_glm_flash_indexer import CudaFlashIndexer
    from damendaemon.cuda_prepared_projection import CudaPackedProjection
    from damendaemon.glm_q4 import decode_blockwise_fp8
    from damendaemon.quantization import dequantize_q4, quantize_q4

    if torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(int(12 * 2**30) / properties.total_memory)
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  layer=args.layer, tokens=args.tokens, model_qualified=False, gates=[])
    save(args.output, report)
    try:
        device = torch.device('cuda:0')
        layer = args.layer
        with open(f'{args.model}/config.json') as reader:
            text = json.load(reader)['text_config']
        heads = text['num_attention_heads']
        nope = text['qk_nope_head_dim']
        vdim = text['v_head_dim']
        epsilon = text['rms_norm_eps']
        geometry = dict(heads=heads, nope=nope, v=vdim, qr=text['q_lora_rank'],
                        kr=text['kv_lora_rank'], index_heads=text['index_n_heads'],
                        index_width=text['index_head_dim'], pool=text['index_kpool'],
                        top_k=text['index_topk'])
        if args.tokens < 1 or args.tokens > 1024:
            raise ValueError('cell tokens out of bounds')

        index = json.load(open(f'{args.model}/model.safetensors.index.json'))['weight_map']
        attn = f'model.layers.{layer}.self_attn'
        packed_names = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'o_proj']
        dense_names = ['q_a_layernorm', 'kv_a_layernorm', 'kv_b_proj']
        # Checkpoint naming: matrix projections carry .weight; the pool APE
        # and gate parameters ship bare (no .weight suffix).
        indexer_checkpoint = ['indexer.wq_b.weight', 'indexer.wk.weight', 'indexer.weights_proj.weight',
                              'indexer.k_norm.weight', 'indexer.k_norm.bias', 'indexer.index_kpool_compress_ape',
                              'indexer.index_kpool_compress_gate']
        wanted = [f'{attn}.{n}.weight' for n in packed_names + dense_names]
        wanted += [f'{attn}.{n}.weight_scale_inv' for n in packed_names]
        wanted += [f'{attn}.{n}' for n in indexer_checkpoint]
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ValueError(f'layer {layer} is not a sparse-attention layer (misses {missing[:3]})')

        by_shard = {}
        for name in wanted:
            by_shard.setdefault(index[name], []).append(name)
        tensors = {}
        for shard, names in by_shard.items():
            with safe_open(f'{args.model}/{shard}', framework='pt', device='cpu') as f:
                for name in names:
                    tensors[name] = f.get_tensor(name)
        report.update(shards=len(by_shard), **geometry)
        save(args.output, report)

        def packed_pair(name):
            raw = tensors[f'{attn}.{name}.weight']
            if raw.dtype == torch.uint8:
                codes = raw
            elif raw.dtype == torch.float8_e4m3fn:
                codes = raw.view(torch.uint8)
            else:
                raise ValueError(f'unexpected shipped codes dtype {raw.dtype} for {name}')
            scales = tensors[f'{attn}.{name}.weight_scale_inv']
            if scales.dtype != torch.float32:
                raise ValueError(f'unexpected shipped scale dtype {scales.dtype} for {name}')
            return codes, scales

        oracle_projections = {}
        q4_projections = {}
        weight_drift = []
        for name in packed_names:
            codes, scales = packed_pair(name)
            truth = decode_blockwise_fp8(codes, scales)
            qweight, qscales = quantize_q4(truth)
            approximate = dequantize_q4(qweight, qscales)
            weight_drift.append(dict(projection=name,
                                     rel_rms=float(((approximate - truth).norm() / truth.norm()).item())))
            oracle_projections[name + '.weight'] = CudaPackedProjection(
                codes.to(device), scales.to(device), format='e4m3fn_f32', dtype=torch.float32)
            q4_projections[name + '.weight'] = CudaPackedProjection(
                qweight.to(device), qscales.to(device), format='q4_0_f16', dtype=torch.float32)
        report['weight_drift'] = weight_drift
        fp8_bytes = sum(tensors[f'{attn}.{n}.weight'].numel()
                        + tensors[f'{attn}.{n}.weight_scale_inv'].numel() * 4 for n in packed_names)
        q4_bytes = sum(p.weight.numel() + p.scales.numel() * 2
                       for p in q4_projections.values())
        report.update(fp8_projection_bytes=fp8_bytes, q4_projection_bytes=q4_bytes)
        save(args.output, report)

        # Dense parts ship floating and are shared by reference: kv_b_proj,
        # both layernorms, the whole indexer — identical objects in both runs.
        dense = {}
        for name in dense_names:
            dense[name + '.weight'] = tensors[f'{attn}.{name}.weight'].float().to(device)
        indexer_weights = {}
        for name in ('wq_b.weight', 'wk.weight', 'weights_proj.weight', 'k_norm.weight',
                     'k_norm.bias', 'index_kpool_compress_ape', 'index_kpool_compress_gate'):
            indexer_weights[name] = tensors[f'{attn}.indexer.{name}'].float().to(device)
        budgets = dict(max_workspace_bytes=512 * 1024**2, max_output_bytes=128 * 1024**2)
        indexer = CudaFlashIndexer(indexer_weights, heads=geometry['index_heads'],
                                    pool_size=geometry['pool'], top_k=geometry['top_k'], **budgets)

        def build(projections):
            weights = dict(projections)
            weights.update(dense)
            return CudaFlashSparseAttention(weights, indexer, heads=heads, head_dim=nope,
                                            value_dim=vdim, epsilon=epsilon, **budgets)

        oracle = build(oracle_projections)
        candidate = build(q4_projections)

        generator = torch.Generator().manual_seed(719)
        hidden = torch.randn(1, args.tokens, 6144, generator=generator).to(device)
        state = oracle.new_state(1, args.tokens)
        oracle_out, oracle_indices = oracle.forward(hidden, state, position=0)
        state = candidate.new_state(1, args.tokens)
        candidate_out, candidate_indices = candidate.forward(hidden, state, position=0)

        report['peak_reserved_bytes'] = torch.cuda.max_memory_allocated()
        delta = candidate_out - oracle_out
        rel = float((delta.norm() / oracle_out.norm()).item())
        max_abs = float(delta.abs().max().item())
        absmax = float(oracle_out.abs().max().item())
        parity = dict(gate='attention_parity', rel_rms=rel, max_abs=max_abs,
                      oracle_absmax=absmax, max_abs_of_absmax=max_abs / absmax)
        parity['passed'] = rel <= 0.15 and max_abs <= 0.15 * absmax
        report['gates'].append(parity)
        save(args.output, report)

        selection = dict(gate='selection_exactness',
                         identical=bool(torch.equal(oracle_indices, candidate_indices)),
                         selected_per_query=oracle_indices.shape[-1])
        selection['passed'] = selection['identical']
        report['gates'].append(selection)
        report['gates'].append(dict(gate='finiteness',
                                   passed=bool(torch.isfinite(candidate_out).all()
                                               and torch.isfinite(oracle_out).all())))
        report['gates'].append(dict(gate='memory', fp8_bytes=fp8_bytes, q4_bytes=q4_bytes,
                                   passed=q4_bytes < fp8_bytes))
        report['passed'] = all(g['passed'] for g in report['gates'])
        report['model_qualified'] = True
        save(args.output, report)
        print('RUNG4B_CELL_DONE', 'passed' if report['passed'] else 'FAILED')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
