"""Replace tokenizer metadata for a validation fixture without changing weights.

The original file is never modified. Requires identical vocabulary IDs/text and
writes a per-tensor equality report after reopening the output. Not a runtime
repair or a claim that the original tokenizer was equivalent.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('vocabulary')
    parser.add_argument('--output', required=True); parser.add_argument('--report', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():raise ValueError('output must be a new file')
    source, vocab = GGUFReader(args.source), GGUFReader(args.vocabulary)
    if source.fields['tokenizer.ggml.tokens'].contents() != vocab.fields['tokenizer.ggml.tokens'].contents():
        raise ValueError('vocabulary IDs/text must match before replacing metadata')
    fields = dict(source.fields)
    fields.update({k:v for k,v in vocab.fields.items() if k.startswith('tokenizer.ggml.')})
    writer = GGUFWriter(str(destination), source.fields['general.architecture'].contents(), endianess=source.endianess)
    for key, field in fields.items():
        if key.startswith('GGUF.') or key == 'general.architecture':continue
        kind = field.types[0]
        writer.add_key_value(key, field.contents(), kind, field.types[-1] if kind == GGUFValueType.ARRAY else None)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    restored = GGUFReader(str(destination))
    checks = [dict(name=a.name, equal=a.name == b.name and a.tensor_type == b.tensor_type
                   and np.array_equal(a.shape,b.shape) and np.array_equal(a.data,b.data))
              for a,b in zip(source.tensors, restored.tensors)]
    passed = len(source.tensors) == len(restored.tensors) and all(c['equal'] for c in checks)
    report = dict(passed=passed, source_sha256=digest(args.source), vocabulary_sha256=digest(args.vocabulary),
                  output_sha256=digest(destination), changed_metadata=[k for k in fields if k.startswith('tokenizer.ggml.')
                    and (k not in source.fields or source.fields[k].contents()!=fields[k].contents())], tensors=checks)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tensors'}))
    if not passed:raise AssertionError('tensor bytes changed')


if __name__ == '__main__':main()
