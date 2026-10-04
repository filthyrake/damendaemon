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


def fixture(path, kind=8, arch='qwen2', tied=False, packed_embedding=False, mixed=False, rope_factors=None, causal=True):
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
    c = dict(model_type=arch, hidden_size=32, num_attention_heads=4, head_dim=8,
             num_key_value_heads=2, intermediate_size=64, num_hidden_layers=1, vocab_size=len(tokens))
    writer = GGUFWriter(str(path), arch)
    for key, value in [('context_length', 128), ('embedding_length', 32), ('feed_forward_length', 64),
                       ('block_count', 1), ('attention.head_count', 4), ('attention.head_count_kv', 2)]:
        writer.add_uint32(f'{arch}.{key}', value)
    writer.add_float32(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)
    writer.add_bool(f'{arch}.attention.causal', causal)
    writer.add_tokenizer_model('gpt2'); writer.add_tokenizer_pre('qwen2')
    # Use one merge; empty metadata arrays have no subtype in upstream ReaderField.
    tokens.append('ab')
    writer.add_token_list(tokens); writer.add_token_types([1] * 256 + [3, 3, 3, 1])
    c['vocab_size'] = len(tokens)
    writer.add_token_merges(['a b']); writer.add_eos_token_id(258)
    writer.add_chat_template("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    rng = np.random.default_rng(7)
    names = {} if tied else {'output.weight': True}
    if rope_factors is not None:names['rope_freqs.weight'] = True
    arrays = {}
    for _, group in tensor_groups(c, names):
        for name, shape in group:
            value = (np.asarray(rope_factors, dtype=np.float32) if name == 'rope_freqs.weight'
                     else rng.normal(0, .1, shape).astype(np.float32))
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
                    self.assertEqual(model.weights['embed'].dtype, torch.uint8 if kind==2 else torch.int8)
                    self.assertIn('embed', model.scales)
                    if tied:
                        self.assertIs(model.weights['embed'], model.weights['head'])
                        self.assertIs(model.scales['embed'], model.scales['head'])
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


    def test_embedding_storage_estimate_tracks_backend_without_stale_plan(self):
        from unittest.mock import patch
        from dllm.quantization import dequantize
        for tied in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.gguf'; fixture(path, tied=tied, packed_embedding=True)
                with patch('dllm.gguf_model.selected_device', return_value='cuda'):
                    packed = GgufWeights(path, 64, 'none')
                    self.assertEqual(packed.bytes, weight_bytes(path, 'none'))
                with patch('dllm.gguf_model.selected_device', return_value='mps'):
                    expanded = GgufWeights(path, 64, 'none')
                    self.assertEqual(expanded.bytes, weight_bytes(path, 'none'))
                self.assertLess(packed.bytes, expanded.bytes)
                self.assertTrue(torch.equal(expanded.weights['embed'],
                    dequantize(packed.weights['embed'], packed.scales['embed'], torch.float16)))
                self.assertNotIn('embed', expanded.scales)

    def test_packed_embedding_extreme_scale_checks_actual_values(self):
        from unittest.mock import patch
        from dllm.gguf_model import read_tensor
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, packed_embedding=True)
            def altered(file, info, nonzero=False):
                data, scales = read_tensor(file, info)
                if info.name == 'token_embd.weight':
                    data.zero_(); data[0,0] = 127 if nonzero else 0
                    scales.fill_(65504)
                return data, scales
            with patch('dllm.gguf_model.read_tensor', side_effect=altered):
                GgufWeights(path, 64, 'none')  # Large scales with zero values are finite.
            with patch('dllm.gguf_model.read_tensor', side_effect=lambda f,i:altered(f,i,True)):
                with self.assertRaisesRegex(ValueError, 'finite FP16'):
                    GgufWeights(path, 64, 'none')


    def test_llama_restores_packed_qk_rows_and_preserves_rotary_factors(self):
        from dllm.quantization import dequantize as q8, dequantize_q4 as q4
        factors = [1.0, 1.000123, 2.7182818, 32.0]
        for kind in (0, 2, 8):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.gguf'
                fixture(path, kind, arch='llama', rope_factors=factors)
                model = GgufWeights(path, 64, 'none')
                self.assertEqual(model.bytes, weight_bytes(path, 'none'))
                self.assertEqual(model.weights['rope_factors'].dtype, torch.float32)
                np.testing.assert_array_equal(model.weights['rope_factors'].numpy(), np.array(factors,dtype=np.float32))
                reader = GGUFReader(path)
                original = {t.name:dequantize(t.data,t.tensor_type) for t in reader.tensors}
                values = model.weights['0.qkv.weight']
                if kind in (2,8):values = (q4 if kind==2 else q8)(values,model.scales['0.qkv.weight'])
                else:values=values.float()
                for first,rows,heads,name in ((0,32,4,'q'),(32,16,2,'k')):
                    restored = values[first:first+rows]
                    # Apply the reference's forward permutation; it must recover
                    # the original independent GGUF dequantized row sequence.
                    encoded = restored.reshape(heads,2,rows//heads//2,32).transpose(1,2).reshape(rows,32)
                    expected = original[f'blk.0.attn_{name}.weight']
                    if kind==0:expected=expected.astype(np.float16).astype(np.float32)
                    np.testing.assert_array_equal(encoded.numpy(),expected)
                self.assertNotIn('0.q_norm',model.weights)

    def test_llama_invalid_rotary_factors_are_rejected(self):
        for factors in ([1,2,0,4],[1,2,-1,4],[1,2,float('nan'),4]):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.gguf'
                fixture(path,arch='llama',rope_factors=factors)
                with self.assertRaises(ValueError):GgufWeights(path,64,'none')

    def test_noncausal_decoder_is_rejected_during_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.gguf'; fixture(path,arch='llama',causal=False)
            with self.assertRaisesRegex(ValueError,'causal attention'):plan(path)


if __name__ == '__main__': unittest.main()
