import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

HAS_GGUF = importlib.util.find_spec('gguf') is not None
if HAS_GGUF:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    from gguf.quants import quantize, dequantize
    from dllm.gguf_reader import BoundedReader
    from dllm.gguf_model import GgufWeights, plan, tensor_groups, weight_bytes


def fixture(path, kind=8, arch='qwen2', tied=False, packed_embedding=False, mixed=False):
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
    c = dict(model_type=arch, hidden_size=32, num_attention_heads=4, head_dim=8,
             num_key_value_heads=2, intermediate_size=64, num_hidden_layers=1, vocab_size=len(tokens))
    writer = GGUFWriter(str(path), arch)
    for key, value in [('context_length', 128), ('embedding_length', 32), ('feed_forward_length', 64),
                       ('block_count', 1), ('attention.head_count', 4), ('attention.head_count_kv', 2)]:
        writer.add_uint32(f'{arch}.{key}', value)
    writer.add_float32(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)
    writer.add_tokenizer_model('gpt2'); writer.add_tokenizer_pre('qwen2')
    # Use one merge; empty metadata arrays have no subtype in upstream ReaderField.
    tokens.append('ab')
    writer.add_token_list(tokens); writer.add_token_types([1] * 256 + [3, 3, 3, 1])
    c['vocab_size'] = len(tokens)
    writer.add_token_merges(['a b']); writer.add_eos_token_id(258)
    writer.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    rng = np.random.default_rng(7)
    names = {} if tied else {'output.weight': True}
    arrays = {}
    for _, group in tensor_groups(c, names):
        for name, shape in group:
            value = rng.normal(0, .1, shape).astype(np.float32)
            dtype = kind if len(shape) == 2 and (name != 'token_embd.weight' or packed_embedding) else 0
            if mixed and name.endswith('attn_k.weight'):
                dtype = 2 if kind == 8 else 8
            array = quantize(value, GGMLQuantizationType(dtype))
            writer.add_tensor(name, array, raw_dtype=GGMLQuantizationType(dtype))
            arrays[name] = (value, dtype)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return arrays


@unittest.skipUnless(HAS_GGUF, 'optional gguf dependency is not installed')
class GgufTests(unittest.TestCase):
    def test_native_q4_q8_import_matches_independent_dequantization(self):
        from dllm.quantization import dequantize as q8, dequantize_q4 as q4
        for kind in (0, 1, 2, 8):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.gguf'; fixture(path, kind)
                model = GgufWeights(path, 64, quantization='none')
                self.assertEqual(model.bytes, weight_bytes(path, 'none'))
                reader = GGUFReader(path)
                tensors = {t.name: dequantize(t.data, t.tensor_type) for t in reader.tensors}
                for target, group in tensor_groups(model.config, tensors):
                    expected = np.concatenate([tensors[n] for n, _ in group], axis=0)
                    actual = model.weights[target]
                    if target in model.scales:
                        actual = (q4 if kind == 2 else q8)(actual, model.scales[target])
                    else:
                        expected = expected.astype(np.float16)
                    np.testing.assert_array_equal(actual.numpy(), expected)

    def test_packed_embedding_and_tied_head_storage(self):
        for kind in (2, 8):
            for tied in (False, True):
                with self.subTest(kind=kind, tied=tied), tempfile.TemporaryDirectory() as folder:
                    path = Path(folder)/'model.gguf'; fixture(path, kind, tied=tied, packed_embedding=True)
                    model = GgufWeights(path, 64, 'none')
                    self.assertEqual(model.weights['embed'].dtype, torch.float16)
                    self.assertNotIn('embed', model.scales)
                    self.assertIn('head', model.scales)
                    self.assertEqual(model.bytes, weight_bytes(path, 'none'))

    def test_qwen3_config_qk_norm_and_shared_chat_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, arch='qwen3')
            from dllm.models import config, load
            from dllm.cli import resolve_model
            self.assertEqual(resolve_model(str(path)), path.resolve())
            self.assertEqual(config(path)['model_type'], 'qwen3')
            model = load(path, 64, 'none')
            self.assertEqual(model.weights['0.q_norm'].shape, (8,))
            self.assertEqual(model.chat_tokens([{'role':'user','content':'ab'}]), model.tokenizer.encode('ab').ids)
            self.assertEqual(model.tokenizer.decode(model.tokenizer.encode('café e\u0301 🦊\n').ids), 'café e\u0301 🦊\n')
            self.assertEqual(model.eos, {256, 258})
            with self.assertRaisesRegex(ValueError, 'conversion'):
                weight_bytes(path, 'q8_0')

    def test_incompatible_fused_formats_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, mixed=True)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                plan(path)

    def test_cache_blob_symlinks_remain_gguf_after_resolution(self):
        from dllm.cli import resolve_model
        from dllm.models import config, load, is_gguf
        with tempfile.TemporaryDirectory() as folder:
            blob = Path(folder)/'hash-without-extension'; fixture(blob)
            link = Path(folder)/'model.gguf'; link.symlink_to(blob)
            resolved = resolve_model(str(link))
            self.assertEqual(resolved, blob.resolve())
            self.assertTrue(is_gguf(resolved))
            self.assertEqual(config(resolved)['model_type'], 'qwen2')
            self.assertEqual(load(resolved, 64, 'none').weight_format, 'gguf_q8_0')
            other = Path(folder)/'unknown'; other.write_bytes(b'not a model')
            self.assertFalse(is_gguf(other))

    def test_cache_invalidates_when_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, kind=8)
            first = plan(path)
            self.assertIs(first, plan(path))
            fixture(path, kind=2)
            self.assertEqual(plan(path).quantization, 'q4_0')
            self.assertNotEqual(first.identity, plan(path).identity)

    def test_truncated_invalid_counts_and_overlapping_data_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path)
            original = path.read_bytes()
            for data in (b'GGUF', original[:200], original[:-40],
                         original[:8] + struct.pack('<Q', 2**63) + original[16:]):
                path.write_bytes(data)
                with self.assertRaises(ValueError): BoundedReader(path)
            path.write_bytes(original)
            reader = GGUFReader(path)
            offset_field = reader.tensors[1].field
            offset = offset_field.offset + sum(p.nbytes for p in offset_field.parts[:-1])
            del reader
            data = bytearray(original); struct.pack_into('<Q', data, offset, 0)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'overlap'): BoundedReader(path)

    def test_oversized_metadata_array_and_nonfinite_scale_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path)
            original = path.read_bytes(); reader = GGUFReader(path)
            field = reader.fields['tokenizer.ggml.tokens']
            # key length, key data, value type, array subtype, array length
            count_offset = field.offset + sum(p.nbytes for p in field.parts[:4])
            tensor = next(t for t in reader.tensors if int(t.tensor_type) == 8)
            weight_offset = int(tensor.data_offset)
            del reader
            data = bytearray(original); struct.pack_into('<Q', data, count_offset, 2**63)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'oversized'): BoundedReader(path)
            data = bytearray(original); struct.pack_into('<H', data, weight_offset, 0x7c00)
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'non-finite scales'): GgufWeights(path, 64, 'none')


if __name__ == '__main__': unittest.main()
