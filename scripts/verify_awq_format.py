"""Inspect real AWQ tensors against scalar, unsigned-word coefficient decoding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from damendaemon.awq import Matrix, group_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = Path(args.model)
    config = json.loads((path/'config.json').read_text())
    group = group_size(config['quantization_config'])
    if config['model_type'] != 'qwen2':
        raise ValueError('initial checkpoint format verifier requires Qwen2')
    width = config['hidden_size']
    dim = config.get('head_dim', width // config['num_attention_heads'])
    geometry = dict(q_proj=(width, config['num_attention_heads']*dim),
                    k_proj=(width, config['num_key_value_heads']*dim),
                    v_proj=(width, config['num_key_value_heads']*dim),
                    o_proj=(config['num_attention_heads']*dim, width),
                    gate_proj=(width, config['intermediate_size']),
                    up_proj=(width, config['intermediate_size']),
                    down_proj=(config['intermediate_size'], width))
    weight_path = path/'model.safetensors'
    with weight_path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(passed=False, model=str(path.resolve()), revision=path.name,
                  model_sha256=checksum, config=config['quantization_config'], matrices=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  codec_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'damendaemon/awq.py').read_bytes()).hexdigest(),
                  scope='host format and scalar coefficient checks; no model execution or performance qualification')
    try:
        with safe_open(str(weight_path), framework='pt', device='cpu') as source:
            packed_names = {name for name in source.keys() if name.endswith(('.qweight', '.qzeros', '.scales'))}
            for layer in range(config['num_hidden_layers']):
                for projection, (inputs, outputs) in geometry.items():
                    parent = 'self_attn' if projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
                    prefix = f'model.layers.{layer}.{parent}.{projection}'
                    names = [prefix + suffix for suffix in ('.qweight', '.qzeros', '.scales')]
                    matrix = Matrix(*(source.get_tensor(name) for name in names), inputs, outputs, group)
                    packed_names.difference_update(names)
                    decoded = matrix.dequantize().numpy()
                    weight, zeros, scales = (t.numpy() for t in (matrix.weight, matrix.zeros, matrix.scales))
                    positions = sorted({0, inputs-1, *range(group-1, inputs, group), *range(group, inputs, group)})
                    channels = sorted({*range(min(9, outputs)), outputs-1})
                    checked = 0
                    for i in positions:
                        for j in channels:
                            # Independent scalar indexing, rather than the runtime's
                            # broadcasted shift table and expanded group tensors.
                            shift = ((j % 8)//2 + (j % 2)*4)*4
                            value = ((int(weight[i, j//8]) & 0xffffffff) >> shift) & 15
                            zero = ((int(zeros[i//group, j//8]) & 0xffffffff) >> shift) & 15
                            expected = np.float16(np.float32(value-zero) * np.float32(scales[i//group, j]))
                            if decoded[j, i].view(np.uint16) != expected.view(np.uint16):
                                raise AssertionError(f'AWQ coefficient mismatch at {prefix} input={i} output={j}')
                            checked += 1
                    if not np.isfinite(decoded).all():
                        raise AssertionError(f'nonfinite expanded coefficient at {prefix}')
                    report['matrices'].append(dict(name=prefix, inputs=inputs, outputs=outputs,
                                                   packed_bytes=matrix.bytes, scalar_checks=checked, passed=True))
            if packed_names:
                raise ValueError(f'unaccounted packed tensors: {sorted(packed_names)}')
            if config.get('tie_word_embeddings') and 'lm_head.weight' in source.keys():
                report['tied_head_equal'] = torch.equal(source.get_tensor('lm_head.weight'),
                                                        source.get_tensor('model.embed_tokens.weight'))
                if not report['tied_head_equal']:
                    raise AssertionError('declared tied head differs from embedding')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, matrices=len(report['matrices']),
                          scalar_checks=sum(m['scalar_checks'] for m in report['matrices']),
                          tied_head_equal=report.get('tied_head_equal'))))


if __name__ == '__main__':
    main()
