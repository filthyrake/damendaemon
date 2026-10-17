"""Match damendaemon's Q8 output head while retaining a tied checkpoint's FP16 embedding.

Only creates a new validation artifact. Upstream code and source GGUF stay
unchanged. All original tensor bytes must survive the conversion exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
from gguf.quants import quantize


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('--output', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('output must be a new artifact')
    source = GGUFReader(args.source)
    if source.fields['general.architecture'].contents() != 'qwen35':
        raise ValueError('this qualification adapter targets dense Qwen3.5')
    tensors = {t.name: t for t in source.tensors}
    if len(tensors) != len(source.tensors) or 'output.weight' in tensors:
        raise ValueError('expected unique tensors and an implicit tied output head')
    embedding = tensors['token_embd.weight']
    if embedding.tensor_type != GGMLQuantizationType.F16:
        raise ValueError('input embeddings must retain FP16 storage')
    head = quantize(embedding.data.astype(np.float32), GGMLQuantizationType.Q8_0)
    writer = GGUFWriter(str(destination), 'qwen35', endianess=source.endianess)
    for key, field in source.fields.items():
        if key.startswith('GGUF.') or key == 'general.architecture':
            continue
        kind = field.types[0]
        writer.add_key_value(key, field.contents(), kind,
                             field.types[-1] if kind == GGUFValueType.ARRAY else None)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.add_tensor('output.weight', head, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    restored = GGUFReader(str(destination))
    copies = {t.name: t for t in restored.tensors}
    checks = []
    for name, original in tensors.items():
        copy = copies[name]
        checks.append(dict(name=name, equal=copy.tensor_type == original.tensor_type
                           and np.array_equal(copy.shape, original.shape)
                           and np.array_equal(copy.data, original.data)))
    output = copies['output.weight']
    head_passed = (output.tensor_type == GGMLQuantizationType.Q8_0
                   and np.array_equal(output.shape, embedding.shape)
                   and np.array_equal(output.data, head))
    passed = (len(copies) == len(tensors)+1 and head_passed and all(c['equal'] for c in checks))
    report = dict(passed=passed, source_sha256=digest(args.source), output_sha256=digest(destination),
                  output_head_q8_verified=head_passed, unchanged_tensors=checks,
                  scope='explicit Q8 output head, unchanged FP16 input embedding and all other tensors')
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise AssertionError('comparator coefficient preservation failed')


if __name__ == '__main__':
    main()
