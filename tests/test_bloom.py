import unittest
import torch
from dllm.bloom import alibi_slopes, load_weights
from dllm.config import decoder_config


def config(**changes):
    return dict(model_type='bloom', architectures=['BloomForCausalLM'], n_embed=16,
                num_attention_heads=2, n_layer=1, vocab_size=32, **changes)


class BloomTests(unittest.TestCase):
    def test_configuration(self):
        c = decoder_config(config())
        self.assertEqual((c['head_dim'], c['intermediate_size'], c['max_position_embeddings']), (8,64,2048))
        for fields in ({'hidden_size':32}, {'pretraining_tp':True}, {'n_inner':32}, {'pretraining_tp':2}, {'tie_word_embeddings':False},
                       {'layer_norm_epsilon':float('nan')}, {'apply_residual_connection_post_layernorm':1},
                       {'rope_scaling':{'type':'linear','factor':2}}, {'quantization_config':{'bits':4}}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                decoder_config(config(**fields))

    def test_non_power_of_two_alibi_head_order(self):
        # Eight geometric slopes plus odd-indexed slopes from a sixteen-head grid.
        expected = [2.**(-i) for i in range(1,9)] + [2.**(-i/2) for i in (1,3,5,7)]
        torch.testing.assert_close(alibi_slopes(12), torch.tensor(expected), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(alibi_slopes(1), torch.tensor([1/256]))

    def test_qkv_head_order_and_tied_embedding(self):
        c = decoder_config(config())
        state = {'word_embeddings.weight':torch.zeros(32,16)}
        for name in ('word_embeddings_layernorm','ln_f','h.0.input_layernorm','h.0.post_attention_layernorm'):
            state[name+'.weight'] = torch.ones(16)
            state[name+'.bias'] = torch.zeros(16)
        for name, shape in (('self_attention.query_key_value',(48,16)), ('self_attention.dense',(16,16)),
                            ('mlp.dense_h_to_4h',(64,16)), ('mlp.dense_4h_to_h',(16,64))):
            state['h.0.'+name+'.weight'] = torch.arange(shape[0]).float()[:,None].expand(shape).clone()
            state['h.0.'+name+'.bias'] = torch.arange(shape[0]).float()
        weights = load_weights(state,c)
        expected = [*range(0,8),*range(24,32),*range(8,16),*range(32,40),*range(16,24),*range(40,48)]
        self.assertEqual(weights['0.qkv.weight'][:,0].tolist(), expected)
        self.assertEqual(weights['0.qkv.bias'].tolist(), expected)
        self.assertIs(weights['embed'], weights['head'])
