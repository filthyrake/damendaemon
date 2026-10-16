import unittest
import torch
from dllm.config import decoder_config
from dllm.neox import load_weights


def config(**changes):
    return {'model_type':'gpt_neox','architectures':['GPTNeoXForCausalLM'],
            'hidden_size':16,'intermediate_size':32,'num_attention_heads':2,
            'num_hidden_layers':1,'max_position_embeddings':32,'vocab_size':32,
            'hidden_act':'gelu','rotary_pct':.25,**changes}


class NeoxTests(unittest.TestCase):
    def test_geometry_and_rejected_variants(self):
        parsed=decoder_config(config())
        self.assertEqual(parsed['rotary_dim'],2)
        self.assertEqual(parsed['num_key_value_heads'],2)
        self.assertTrue(parsed['use_parallel_residual'])
        self.assertFalse(decoder_config(config(use_parallel_residual=False))['use_parallel_residual'])
        for changes in ({'rotary_pct':.1},{'rotary_pct':True},{'rotary_pct':float('nan')},
                        {'hidden_act':'gelu_new'},{'use_parallel_residual':1},
                        {'tie_word_embeddings':True},{'attention_bias':False},
                        {'layer_norm_eps':0},{'rotary_emb_base':float('inf')},
                        {'rope_scaling':{'type':'linear','factor':2}},
                        {'quantization_config':{'bits':4}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                decoder_config(config(**changes))

    def test_rotary_head_dimensions_must_fit_host_numeric_range(self):
        for fraction in (.25,1.0,1):
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):
                decoder_config(config(hidden_size=10**400,rotary_pct=fraction))

    def test_interleaved_qkv_is_reordered_by_head(self):
        c=decoder_config(config())
        state={'gpt_neox.embed_in.weight':torch.zeros(32,16),'embed_out.weight':torch.ones(32,16),
               'gpt_neox.final_layer_norm.weight':torch.ones(16),'gpt_neox.final_layer_norm.bias':torch.zeros(16)}
        prefix='gpt_neox.layers.0.'
        for name in ('input_layernorm','post_attention_layernorm'):
            state[prefix+name+'.weight']=torch.ones(16)
            state[prefix+name+'.bias']=torch.zeros(16)
        for name,shape in (('attention.query_key_value',(48,16)),('attention.dense',(16,16)),
                           ('mlp.dense_h_to_4h',(32,16)),('mlp.dense_4h_to_h',(16,32))):
            state[prefix+name+'.weight']=torch.arange(shape[0]).float()[:,None].expand(shape).clone()
            state[prefix+name+'.bias']=torch.arange(shape[0]).float()
        weights=load_weights(state,c)
        expected=[*range(0,8),*range(24,32),*range(8,16),*range(32,40),*range(16,24),*range(40,48)]
        self.assertEqual(weights['0.qkv.weight'][:,0].tolist(),expected)
        self.assertEqual(weights['0.qkv.bias'].tolist(),expected)
        self.assertTrue(weights['0.qkv.weight'].is_contiguous())
