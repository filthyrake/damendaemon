"""Benchmark-only GGUF conversion matching dllm's quantized matrix storage policy.

Requires upstream gguf-py on PYTHONPATH. Input must be an unquantized GGUF
converted from the same pinned checkpoint; this is not a dllm model importer.
"""
import argparse
from pathlib import Path
import numpy as np
import gguf
from gguf.quants import quantize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--format', choices=('q8_0', 'q4_0'), default='q8_0')
    args = parser.parse_args()
    quantization_type = getattr(gguf.GGMLQuantizationType, args.format.upper())
    if args.destination.exists():
        parser.error('destination already exists')
    reader = gguf.GGUFReader(args.source)
    for tensor in reader.tensors:
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            parser.error('source must contain only unquantized F16/F32 tensors')
    writer = gguf.GGUFWriter(args.destination,
        arch=reader.fields['general.architecture'].contents(), endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith('GGUF.') or name in ('general.architecture', 'general.file_type', 'general.quantization_version'):
            continue
        kind = field.types[0]
        writer.add_key_value(name, field.contents(), kind,
            sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    writer.add_file_type(getattr(gguf.LlamaFileType, 'MOSTLY_'+args.format.upper()))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    if 'output.weight' not in tensors:
        tensors['output.weight'] = tensors['token_embd.weight']
    for name, data in tensors.items():
        if data.ndim == 2 and name != 'token_embd.weight':
            # dllm loads projection matrices into FP16 before quantizing.
            data = quantize(data.astype(np.float16).astype(np.float32), quantization_type)
            writer.add_tensor(name, data, raw_dtype=quantization_type)
        else:
            writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == '__main__':
    main()
