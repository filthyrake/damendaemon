"""Offline tensor-header capacity estimate; never a load or performance proof.

Inputs are a pinned inventory and dataclasses.asdict(get_safetensors_metadata()).
Run the actual hybrid weight mapping on shape-only meta tensors, without weights.
"""
import argparse
import gzip
import json
from pathlib import Path

import torch
from dllm.config import decoder_config
from dllm.hybrid import load_weights


def estimate(inventory, metadata):
    config = decoder_config(inventory['config'])
    state = {}
    for shard in metadata['files_metadata'].values():
        for name, tensor in shard['tensors'].items():
            if name in state:
                raise ValueError(f'duplicate tensor: {name}')
            state[name] = torch.empty(tensor['shape'], device='meta')
    weights = load_weights(state, config)
    unexpected = [name for name in state if not name.startswith(('mtp.', 'model.visual.'))]
    if unexpected:
        raise ValueError(f'unconsumed text tensors: {unexpected}')
    totals = {}
    for mode in ('none', 'q8_0', 'q4_0'):
        total = 0
        seen = set()
        for name, tensor in weights.items():
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            n = tensor.numel()
            if mode != 'none' and tensor.ndim == 2 and name != 'embed':
                if tensor.shape[1] % 32:
                    raise ValueError(f'non-packable matrix: {name}')
                total += (n if mode == 'q8_0' else n // 2) + n // 32 * 2
            else:
                total += n * tensor.element_size()
        totals[mode] = total
    full = config['layer_types'].count('full_attention')
    linear = config['num_hidden_layers'] - full
    recurrent = linear * (
        config['linear_num_value_heads'] * config['linear_key_head_dim'] * config['linear_value_head_dim'] * 4
        + (2 * config['linear_num_key_heads'] * config['linear_key_head_dim']
           + config['linear_num_value_heads'] * config['linear_value_head_dim'])
        * config['linear_conv_kernel_dim'] * 2)
    contexts = {}
    for context in (32768, 65536, 131072, 262144):
        state_bytes = recurrent + full * 2 * config['num_key_value_heads'] * context * config['head_dim'] * 2
        contexts[str(context)] = {
            str(users): {'state_bytes': state_bytes * users,
                         'weights_plus_state_bytes': {mode: size + state_bytes * users for mode, size in totals.items()}}
            for users in (1, 4)
        }
    return dict(repository=inventory['repository'], revision=inventory['resolved_revision'],
                weight_bytes=totals, runtime_supported_quantization=['none', 'q8_0'],
                hypothetical_only_quantization=['q4_0'], contexts=contexts, mapped_tensors=len(weights),
                excluded_tensors=len(state),
                limitation='Shape-only text-loader and packed-storage estimate. Excludes workspace, allocator, graphs, communication, host copies and loading peaks. No weight values, inference, quality or performance validated.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inventory', type=Path)
    p.add_argument('metadata', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    opener = gzip.open if args.metadata.suffix == '.gz' else open
    with opener(args.metadata, 'rt') as f:
        metadata = json.load(f)
    result = estimate(json.loads(args.inventory.read_text()), metadata)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['weight_bytes'], indent=2))


if __name__ == '__main__':
    main()
