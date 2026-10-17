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
    from damendaemon.gguf_reader import BoundedReader
    from damendaemon.gguf_model import GgufWeights, plan, tensor_groups, weight_bytes


def fixture(path, kind=8, arch='qwen2', tied=False, packed_embedding=False, mixed=False, rope_factors=None, causal=True, hidden=32, window=None, split_max_tensors=0, small_first_shard=False, scaling=None, legacy_scaling=None):
    from tokenizers.pre_tokenizers import ByteLevel
    tokens = sorted(ByteLevel.alphabet()) + ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
    c = dict(model_type=arch, hidden_size=hidden, num_attention_heads=4, head_dim=hidden//4,
             num_key_value_heads=2, intermediate_size=hidden*2, num_hidden_layers=1, vocab_size=len(tokens))
    writer = GGUFWriter(str(path), arch, split_max_tensors=split_max_tensors, small_first_shard=small_first_shard)
    for key, value in [('context_length', 128), ('embedding_length', hidden), ('feed_forward_length', hidden*2),
                       ('block_count', 1), ('attention.head_count', 4), ('attention.head_count_kv', 2)]:
        writer.add_uint32(f'{arch}.{key}', value)
    writer.add_float32(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)
    writer.add_bool(f'{arch}.attention.causal', causal)
    if scaling is not None:
        writer.add_string(f'{arch}.rope.scaling.type', 'linear')
        writer.add_float32(f'{arch}.rope.scaling.factor', scaling)
    if legacy_scaling is not None:
        writer.add_float32(f'{arch}.rope.scale_linear', legacy_scaling)
    if window is not None:
        writer.add_uint32(f'{arch}.attention.sliding_window', window)
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
                dtype = 14 if kind == 12 else 2 if kind == 8 else 8
            if dtype in (12,13,14):
                from gguf import GGML_QUANT_SIZES
                block,size=GGML_QUANT_SIZES[GGMLQuantizationType(dtype)]
                array=rng.integers(0,256,(value.size//block,size),dtype=np.uint8)
                offset=208 if dtype==14 else 0; count=1 if dtype==14 else 2
                array[:,offset:offset+count*2]=np.full((len(array),count),.001,dtype=np.float16).view(np.uint8)
                array=array.reshape(shape[0],-1)
            else:
                array = quantize(value, GGMLQuantizationType(dtype))
            writer.add_tensor(name, array, raw_dtype=GGMLQuantizationType(dtype))
            arrays[name] = (value, dtype)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    return arrays


@unittest.skipUnless(HAS_GGUF, 'optional gguf dependency is not installed')
class GgufTests(unittest.TestCase):
    def test_native_q4_q8_import_matches_independent_dequantization(self):
        from damendaemon.quantization import dequantize as q8, dequantize_q4 as q4
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
            from damendaemon.models import config, load
            from damendaemon.cli import resolve_model
            self.assertEqual(resolve_model(str(path)), path.resolve())
            self.assertEqual(config(path)['model_type'], 'qwen3')
            model = load(path, 64, 'none')
            self.assertEqual(model.weights['0.q_norm'].shape, (8,))
            self.assertEqual(model.chat_tokens([{'role':'user','content':'ab'}]), model.tokenizer.encode('ab').ids)
            self.assertEqual(model.tokenizer.decode(model.tokenizer.encode('café e\u0301 🦊\n').ids), 'café e\u0301 🦊\n')
            self.assertEqual(model.eos, {256, 258})
            with self.assertRaisesRegex(ValueError, 'conversion'):
                weight_bytes(path, 'q8_0')

    def test_mixed_fused_formats_preserve_raw_segments(self):
        from unittest.mock import patch
        from gguf import GGML_QUANT_SIZES
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, mixed=True)
            model=GgufWeights(path,64,'none')
            width,spec=model.packed_formats['0.qkv.weight']
            self.assertEqual([s[2] for s in spec],[8,2,8])
            expected={t.name:dequantize(t.data,t.tensor_type) for t in GGUFReader(path).tensors}
            actual=[];raw=model.weights['0.qkv.weight'].numpy()
            for first,last,kind,offset in spec:
                block,size=GGML_QUANT_SIZES[GGMLQuantizationType(kind)]
                count=(last-first)*width//block*size
                actual.append(dequantize(raw[offset:offset+count].reshape(-1,size),GGMLQuantizationType(kind)).reshape(last-first,width))
            np.testing.assert_array_equal(np.concatenate(actual),np.concatenate([expected[f'blk.0.attn_{a}.weight'] for a in ('q','k','v')]))
            self.assertEqual(model.bytes,weight_bytes(path,'none'))
            with patch('damendaemon.gguf_model.selected_device',return_value='mps'):
                with self.assertRaisesRegex(ValueError,'require CUDA'):weight_bytes(path,'none')

    def test_cache_blob_symlinks_remain_gguf_after_resolution(self):
        from damendaemon.cli import resolve_model
        from damendaemon.models import config, load, is_gguf
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
        from damendaemon.quantization import dequantize
        for tied in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.gguf'; fixture(path, tied=tied, packed_embedding=True)
                with patch('damendaemon.gguf_model.selected_device', return_value='cuda'):
                    packed = GgufWeights(path, 64, 'none')
                    self.assertEqual(packed.bytes, weight_bytes(path, 'none'))
                with patch('damendaemon.gguf_model.selected_device', return_value='mps'):
                    expanded = GgufWeights(path, 64, 'none')
                    self.assertEqual(expanded.bytes, weight_bytes(path, 'none'))
                self.assertLess(packed.bytes, expanded.bytes)
                self.assertTrue(torch.equal(expanded.weights['embed'],
                    dequantize(packed.weights['embed'], packed.scales['embed'], torch.float16)))
                self.assertNotIn('embed', expanded.scales)

    def test_packed_embedding_extreme_scale_checks_actual_values(self):
        from unittest.mock import patch
        from damendaemon.gguf_model import read_tensor
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.gguf'; fixture(path, packed_embedding=True)
            def altered(file, info, nonzero=False):
                data, scales = read_tensor(file, info)
                if info.name == 'token_embd.weight':
                    data.zero_(); data[0,0] = 127 if nonzero else 0
                    scales.fill_(65504)
                return data, scales
            with patch('damendaemon.gguf_model.read_tensor', side_effect=altered):
                GgufWeights(path, 64, 'none')  # Large scales with zero values are finite.
            with patch('damendaemon.gguf_model.read_tensor', side_effect=lambda f,i:altered(f,i,True)):
                with self.assertRaisesRegex(ValueError, 'finite FP16'):
                    GgufWeights(path, 64, 'none')


    def test_llama_restores_packed_qk_rows_and_preserves_rotary_factors(self):
        from damendaemon.quantization import dequantize as q8, dequantize_q4 as q4
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

    def test_linear_rotary_metadata_preserves_packed_weights_and_context(self):
        from damendaemon.rotary import inverse_frequencies
        for arch in ('qwen2', 'qwen3', 'llama'):
            for factor in (1.0, 4.0):
                with self.subTest(arch=arch, factor=factor), tempfile.TemporaryDirectory() as folder:
                    paths = [Path(folder)/f'{label}.gguf' for label in ('plain', 'scaled')]
                    fixture(paths[0], arch=arch)
                    fixture(paths[1], arch=arch, scaling=factor)
                    plain, scaled = [GgufWeights(path, 128, 'none') for path in paths]
                    self.assertEqual(scaled.config['rope_scaling'], {'rope_type': 'linear', 'factor': factor})
                    self.assertEqual(scaled.config['max_position_embeddings'], 128)
                    self.assertEqual(plain.packed_formats, scaled.packed_formats)
                    for name in plain.weights:
                        self.assertTrue(torch.equal(plain.weights[name], scaled.weights[name]), name)
                    for name in plain.scales:
                        self.assertTrue(torch.equal(plain.scales[name], scaled.scales[name]), name)
                    # llama.cpp uses 1/factor as the static frequency scale.
                    expected = torch.tensor([10000.0 ** (-i/8) / factor for i in range(0, 8, 2)])
                    torch.testing.assert_close(inverse_frequencies(scaled.config, 'cpu'), expected)
                    with self.assertRaises(ValueError):
                        GgufWeights(paths[1], 129, 'none')

    def test_legacy_linear_rotary_metadata_is_not_ignored(self):
        from damendaemon.rotary import inverse_frequencies
        with tempfile.TemporaryDirectory() as folder:
            modern, legacy = [Path(folder)/f'{label}.gguf' for label in ('modern', 'legacy')]
            fixture(modern, arch='llama', scaling=4)
            fixture(legacy, arch='llama', legacy_scaling=4)
            expected, actual = [GgufWeights(path, 128, 'none') for path in (modern, legacy)]
            self.assertEqual(actual.config, expected.config)
            torch.testing.assert_close(inverse_frequencies(actual.config, 'cpu'),
                                       torch.tensor([10000.0 ** (-i/8)/4 for i in range(0, 8, 2)]))
            for name in expected.weights:
                self.assertTrue(torch.equal(expected.weights[name], actual.weights[name]), name)
            for name in expected.scales:
                self.assertTrue(torch.equal(expected.scales[name], actual.scales[name]), name)
            for factor in (0, .5, -1, float('nan'), float('inf')):
                path = Path(folder)/f'invalid-{factor}.gguf'
                fixture(path, legacy_scaling=factor)
                with self.assertRaises(ValueError):
                    plan(path)
            for factor in (2, 4):
                path = Path(folder)/f'ambiguous-{factor}.gguf'
                fixture(path, scaling=4, legacy_scaling=factor)
                with self.assertRaises(ValueError):
                    plan(path)

    def test_invalid_rotary_scaling_metadata_fails_preflight(self):
        from damendaemon.gguf_model import model_config
        cases = [{'type': 'linear', 'factor': factor} for factor in (None, True, '4', 0, .5, -1, float('nan'), float('inf'))]
        cases += [{'type': kind, 'factor': 4} for kind in ('none', 'yarn', 'longrope')]
        cases += [{'type': 'linear', 'factor': 4, 'original_context_length': 32}, {'factor': 4}]
        for scaling in cases:
            with self.subTest(scaling=scaling):
                metadata = {'general.architecture': 'llama', 'llama.embedding_length': 32,
                            'llama.feed_forward_length': 64, 'llama.block_count': 1,
                            'llama.context_length': 128, 'llama.attention.head_count': 4,
                            'llama.attention.head_count_kv': 2, 'llama.attention.layer_norm_rms_epsilon': 1e-6}
                self.assertIsNone(model_config(metadata, {}, 32000, {2})['rope_scaling'])
                metadata.update({f'llama.rope.scaling.{key}': value for key, value in scaling.items()})
                with self.assertRaises(ValueError):
                    model_config(metadata, {}, 32000, {2})

    def test_noncausal_decoder_is_rejected_during_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.gguf'; fixture(path,arch='llama',causal=False)
            with self.assertRaisesRegex(ValueError,'causal attention'):plan(path)

    def test_uniform_window_maps_to_mistral_and_preserves_packed_rotary_rows(self):
        for kind in (0, 2, 8, 12):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                paths = [Path(folder)/f'{label}.gguf' for label in ('full', 'window')]
                for path, window in zip(paths, (0, 64)):
                    fixture(path, kind, arch='llama', hidden=256, window=window, mixed=kind==12)
                full, windowed = [GgufWeights(path, 128, 'none') for path in paths]
                self.assertEqual(full.config['model_type'], 'llama')
                self.assertEqual(windowed.config['model_type'], 'mistral')
                self.assertEqual(windowed.config['sliding_window'], 64)
                self.assertEqual(full.packed_formats, windowed.packed_formats)
                for name in full.weights:
                    self.assertTrue(torch.equal(full.weights[name], windowed.weights[name]), name)
                for name in full.scales:
                    self.assertTrue(torch.equal(full.scales[name], windowed.scales[name]), name)

    def test_invalid_or_nonuniform_window_metadata_fails_preflight(self):
        from damendaemon.gguf_model import model_config
        for arch, value, pattern in (('llama', -1, None), ('llama', True, None),
                                     ('llama', 64.0, None), ('qwen2', 64, None),
                                     ('qwen3', 64, None), ('llama', 64, 4),
                                     ('llama', 0, [True, False])):
            with self.subTest(arch=arch, value=value, pattern=pattern):
                metadata = {'general.architecture': arch, f'{arch}.attention.sliding_window': value}
                if pattern is not None:
                    metadata[f'{arch}.attention.sliding_window_pattern'] = pattern
                with self.assertRaisesRegex(ValueError, 'window'):
                    model_config(metadata, {}, 32000, {2})

    def test_kquant_import_matches_independent_blocks_and_storage(self):
        from gguf import GGML_QUANT_SIZES
        for kind in (12,13,14):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.gguf'
                fixture(path,kind=kind,hidden=256,packed_embedding=True,tied=True,mixed=kind==12)
                model=GgufWeights(path,64,'none')
                expected={t.name:dequantize(t.data,t.tensor_type) for t in GGUFReader(path).tensors}
                for target,group in tensor_groups(model.config,expected):
                    target_values=np.concatenate([expected[name] for name,_ in group])
                    if target not in model.packed_formats:
                        np.testing.assert_array_equal(model.weights[target].numpy(),target_values.astype(np.float16))
                        continue
                    width,spec=model.packed_formats[target];raw=model.weights[target].numpy();actual=[]
                    for first,last,fmt,offset in spec:
                        block,size=GGML_QUANT_SIZES[GGMLQuantizationType(fmt)]
                        count=(last-first)*width//block*size
                        actual.append(dequantize(raw[offset:offset+count].reshape(-1,size),GGMLQuantizationType(fmt)).reshape(last-first,width))
                    np.testing.assert_array_equal(np.concatenate(actual),target_values)
                self.assertEqual(model.bytes,weight_bytes(path,'none'))
                self.assertIs(model.weights['head'],model.weights['embed'])
                self.assertEqual(model.packed_formats['head'],model.packed_formats['embed'])
                self.assertEqual(model.scales,{})


if __name__ == '__main__': unittest.main()
