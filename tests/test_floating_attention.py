import unittest
from unittest.mock import patch

import torch

from dllm.deepseek_position import frequencies, rotate_pairs
from dllm.floating_attention import _query_norm, attention_prefill


def fixture(ratio=0, length=13, dtype=torch.float32, dim=8):
    rng = torch.Generator().manual_seed(113)
    hidden = torch.randn(1, length, dim, generator=rng).to(dtype)
    weights = {}
    for name, shape in {'wq_a.weight':(4,dim), 'wq_b.weight':(2*128,4), 'wkv.weight':(128,dim),
                        'wo_a.weight':(2*3,128), 'wo_b.weight':(dim,2*3)}.items():
        weights[name] = (torch.randn(shape, generator=rng)*.1).to(dtype)
    for name, width in (('q_norm.weight',4), ('kv_norm.weight',128), ('attn_sink',2)):
        weights[name] = torch.randn(width, generator=rng)
    def compressor(prefix, width):
        channels = width*(2 if ratio == 4 else 1)
        for suffix, shape in (('wkv.weight',(channels,dim)), ('wgate.weight',(channels,dim)),
                              ('ape',(ratio,channels)), ('norm.weight',(width,))):
            weights[prefix+'.'+suffix] = torch.randn(shape, generator=rng)*.1
    if ratio:
        compressor('compressor',128)
    if ratio == 4:
        compressor('indexer.compressor',128)
        weights['indexer.wq_b.weight'] = (torch.randn(2*128,4,generator=rng)*.1).to(dtype)
        weights['indexer.weights_proj.weight'] = (torch.randn(2,dim,generator=rng)*.1).to(dtype)
    return hidden, weights, frequencies(64, base=160000 if ratio else 10000)


class FloatingAttentionTests(unittest.TestCase):
    def test_chunk_equivalence_variants_and_compact_cache_state(self):
        for ratio, length in ((0,13),(4,13),(128,130)):
            args = fixture(ratio, length)
            expected, expected_state = attention_prefill(*args, groups=2, window=4, ratio=ratio, index_top_k=2)
            for chunk in (1, 3, 32):
                actual, state = attention_prefill(*args, groups=2, window=4, ratio=ratio, index_top_k=2, query_chunk=chunk)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
                torch.testing.assert_close(state.window, expected_state.window, rtol=3e-5, atol=1e-6)
                self.assertEqual(state.position, length)
                self.assertEqual(state.cache.untyped_storage().nbytes(), 1*(4+(length//ratio if ratio else 0))*128*4)
                self.assertEqual(state.compressed.shape, (1,length//ratio if ratio else 0,128))
                if ratio:
                    torch.testing.assert_close(state.compressed, expected_state.compressed, rtol=3e-5, atol=1e-6)
                    self.assertEqual(state.pool.position, length)
                if ratio == 4:
                    self.assertEqual(state.index_pool.position, length)
                    torch.testing.assert_close(state.index_keys, expected_state.index_keys, rtol=0, atol=0)

    def test_dense_attention_control_and_group_projection_orientation(self):
        args = fixture(4, 13)
        from dllm import floating_attention
        captured = []
        def dense(query, kv, indices, sink, **kwargs):
            output = torch.zeros_like(query)
            for b in range(query.shape[0]):
                for s in range(query.shape[1]):
                    ids = [i for i in indices[b,s].tolist() if i >= 0]
                    for h in range(query.shape[2]):
                        vectors = kv[b, ids].float()
                        logits = (vectors@query[b,s,h].float())*query.shape[-1]**-.5
                        probability = torch.cat((logits,sink[h:h+1])).softmax(0)[:-1]
                        output[b,s,h] = probability@vectors
            captured.append(output.clone())
            return output
        actual, _ = attention_prefill(*args, groups=2, window=4, ratio=4, index_top_k=2)
        with patch('dllm.floating_attention.selected_attention', side_effect=dense):
            expected, _ = attention_prefill(*args, groups=2, window=4, ratio=4, index_top_k=2)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
        # Independently project each head/group instead of reusing the implementation's einsum.
        attended = captured[0]
        pos = torch.arange(13)
        attended[..., -64:] = rotate_pairs(attended[..., -64:], pos, args[2], inverse=True)
        grouped = torch.cat([attended[:,:,g]@args[1]['wo_a.weight'][g*3:(g+1)*3].T for g in range(2)], -1)
        torch.testing.assert_close(expected, grouped@args[1]['wo_b.weight'].T, rtol=3e-5, atol=2e-6)

    def test_low_precision_and_input_immutability(self):
        args = fixture(4, 9, torch.bfloat16)
        hidden, weights, freq = args
        before = {name:tensor.clone() for name,tensor in weights.items()}
        original = hidden.clone()
        output, state = attention_prefill(*args, groups=2, window=4, ratio=4)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(state.window.dtype, torch.bfloat16)
        torch.testing.assert_close(hidden, original)
        for name in weights:
            torch.testing.assert_close(weights[name], before[name])

    def test_contract_rejection_precedes_projection(self):
        hidden, weights, freq = fixture()
        with patch('dllm.floating_attention.linear', side_effect=AssertionError('projection started')):
            for options in ({'groups':3,'window':4}, {'groups':2,'window':0}, {'groups':2,'window':4,'ratio':8}):
                with self.assertRaises(ValueError):
                    attention_prefill(hidden, weights, freq, **options)
            with self.assertRaises(ValueError):
                attention_prefill(hidden, weights | {'unexpected':torch.zeros(1)}, freq, groups=2, window=4)

    def test_overflowing_query_fails_instead_of_silently_normalizing_to_zero(self):
        # A finite FP32 query whose square overflows must not be normalized to
        # zeros through an infinite variance (matches the #470 GLM check class).
        query = torch.zeros(1, 1, 2, 128)
        query[..., 0] = 1e20
        for backend in (None, object()):
            with self.assertRaises(ValueError):
                _query_norm(query, 1e-6, backend)
        torch.testing.assert_close(_query_norm(torch.ones(1, 1, 2, 128), 1e-6, None),
                                   torch.ones(1, 1, 2, 128))
