"""Add pinned HF uniform-window metadata to a new GGUF validation fixture.

Some reference exports omit Mistral's window. This is an explicit semantic
fixture change, not a claim that the original file declared windowed attention.
Every tensor and every other metadata value must remain unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('hf_config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('output must be a new file')
    config = json.loads(Path(args.hf_config).read_text())
    window = config.get('sliding_window')
    if (config.get('model_type') != 'mistral' or type(window) is not int or window <= 0
            or config.get('rope_scaling') or config.get('layer_types')):
        raise ValueError('requires a uniform-window Mistral HF configuration')
    source = GGUFReader(args.source)
    metadata = {name: field.contents() for name, field in source.fields.items()}
    if metadata.get('general.architecture') != 'llama':
        raise ValueError('requires Llama-layout GGUF weights')
    key = 'llama.attention.sliding_window'
    if (metadata.get(key, 0) != 0 or 'llama.attention.sliding_window_pattern' in metadata
            or metadata.get('llama.rope.scaling.type', 'none') != 'none'):
        raise ValueError('source already changes its window or rotary schedule')
    mapping = dict(embedding_length='hidden_size', feed_forward_length='intermediate_size',
                   block_count='num_hidden_layers', context_length='max_position_embeddings',
                   **{'attention.head_count': 'num_attention_heads',
                      'attention.head_count_kv': 'num_key_value_heads',
                      'rope.freq_base': 'rope_theta',
                      'attention.layer_norm_rms_epsilon': 'rms_norm_eps'})
    for field, expected in mapping.items():
        actual = metadata.get('llama.' + field, 10000 if field == 'rope.freq_base' else None)
        if actual is None or not math.isclose(actual, config[expected], rel_tol=1e-6):
            raise ValueError(f'HF/GGUF configuration mismatch: {field}')
    writer = GGUFWriter(str(destination), 'llama', endianess=source.endianess)
    for name, field in source.fields.items():
        if name.startswith('GGUF.') or name in ('general.architecture', key):
            continue
        kind = field.types[0]
        writer.add_key_value(name, field.contents(), kind,
                             field.types[-1] if kind == GGUFValueType.ARRAY else None)
    writer.add_uint32(key, window)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    restored = GGUFReader(str(destination))
    checks = [dict(name=a.name, equal=a.name == b.name and a.tensor_type == b.tensor_type
                   and np.array_equal(a.shape, b.shape) and np.array_equal(a.data, b.data))
              for a, b in zip(source.tensors, restored.tensors)]
    old = {k: v for k, v in metadata.items() if not k.startswith('GGUF.') and k != key}
    new = {k: f.contents() for k, f in restored.fields.items() if not k.startswith('GGUF.') and k != key}
    passed = (old == new and restored.fields[key].contents() == window
              and len(source.tensors) == len(restored.tensors) and all(c['equal'] for c in checks))
    report = dict(passed=passed, source_sha256=digest(args.source), hf_config_sha256=digest(args.hf_config),
                  output_sha256=digest(destination), window=window, changed_metadata=[key],
                  other_metadata_equal=old == new, tensors=checks)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise AssertionError('fixture changed weights or unrelated metadata')


if __name__ == '__main__':
    main()
