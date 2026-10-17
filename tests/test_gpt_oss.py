import math
import unittest
import torch

from damendaemon.gpt_oss import validate_config,load_weights
from damendaemon.rotary import inverse_frequencies,attention_factor


def configuration():
    return {'model_type':'gpt_oss','hidden_size':32,'intermediate_size':32,
        'num_attention_heads':2,'num_key_value_heads':1,'head_dim':64,'num_hidden_layers':2,
        'vocab_size':64,'max_position_embeddings':131072,'hidden_act':'silu','rms_norm_eps':1e-5,
        'rope_theta':150000,'rope_scaling':{'rope_type':'yarn','factor':32,
            'original_max_position_embeddings':4096,'beta_fast':32.,'beta_slow':1.,'truncate':False},
        'quantization_config':{'quant_method':'mxfp4','modules_to_not_convert':[
            'model.layers.*.self_attn','model.layers.*.mlp.router','model.embed_tokens','lm_head']},
        'attention_bias':True,'num_local_experts':4,'num_experts_per_tok':2,'sliding_window':128,
        'layer_types':['sliding_attention','full_attention'],'tie_word_embeddings':False}


def tensors(c):
    h=c['hidden_size'];v=c['vocab_size'];e=c['num_local_experts'];m=c['intermediate_size'];d=c['head_dim']
    s={'model.embed_tokens.weight':torch.ones(v,h,dtype=torch.bfloat16),
       'lm_head.weight':torch.ones(v,h,dtype=torch.bfloat16),'model.norm.weight':torch.ones(h)}
    for i in range(c['num_hidden_layers']):
        root=f'model.layers.{i}.'
        for p,n in (('q',c['num_attention_heads']*d),('k',c['num_key_value_heads']*d),('v',c['num_key_value_heads']*d)):
            s[root+f'self_attn.{p}_proj.weight']=torch.ones(n,h,dtype=torch.bfloat16)
            s[root+f'self_attn.{p}_proj.bias']=torch.ones(n)
        for name,shape in [('self_attn.o_proj.weight',(h,c['num_attention_heads']*d)),
            ('self_attn.o_proj.bias',(h,)),('self_attn.sinks',(c['num_attention_heads'],)),
            ('input_layernorm.weight',(h,)),('post_attention_layernorm.weight',(h,)),
            ('mlp.router.weight',(e,h)),('mlp.router.bias',(e,))]:s[root+name]=torch.ones(shape,dtype=torch.bfloat16)
        for p,n,k in (('gate_up_proj',2*m,h),('down_proj',h,m)):
            name=root+'mlp.experts.'+p
            s[name+'_blocks']=torch.zeros(e,n,k//32,16,dtype=torch.uint8)
            s[name+'_scales']=torch.full((e,n,k//32),127,dtype=torch.uint8)
            s[name+'_bias']=torch.zeros(e,n,dtype=torch.bfloat16)
    return s


class GptOssTests(unittest.TestCase):
    def test_config_rejects_incompatible_layouts(self):
        c=configuration();self.assertEqual(validate_config(c)['swiglu_limit'],7.)
        for change in ({'swiglu_limit':10**400},{'num_experts_per_tok':5},{'experts_per_token':3},{'attention_bias':False},
            {'layer_types':['full_attention']},{'hidden_size':31},{'sliding_window':True},
            {'swiglu_limit':float('nan')},{'quantization_config':{'quant_method':'gptq'}},
            {'rope_scaling':{**c['rope_scaling'],'truncate':True}}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config({**c,**change})

    def test_untruncated_rotary_matches_fractional_boundary_formula(self):
        c=validate_config(configuration());d=c['head_dim'];theta=c['rope_theta']
        low=(d/2)*math.log(4096/(32*2*math.pi))/math.log(theta)
        high=(d/2)*math.log(4096/(2*math.pi))/math.log(theta)
        expected=[]
        for i in range(d//2):
            ramp=min(1,max(0,(i-low)/(high-low)))
            frequency=theta**(2*i/d)
            expected.append((ramp/32+1-ramp)/frequency)
        torch.testing.assert_close(inverse_frequencies(c,'cpu'),torch.tensor(expected),rtol=3e-7,atol=1e-8)
        self.assertAlmostEqual(attention_factor(c),1+.1*math.log(32))
        truncated={**c,'rope_scaling':{**c['rope_scaling'],'truncate':True}}
        self.assertGreater((inverse_frequencies(c,'cpu')-inverse_frequencies(truncated,'cpu')).abs().max(),0)

    def test_weights_remain_packed_and_biases_preserve_precision(self):
        c=validate_config(configuration());state=tensors(c)
        state['model.layers.0.mlp.experts.gate_up_proj_bias'][0,0]=1e6
        # BF16 values outside FP16 range must never silently overflow.
        state['model.layers.0.mlp.router.weight'][0,0]=1e6
        w=load_weights(state,c)
        self.assertFalse(state)
        self.assertEqual(w['0.up.blocks'].dtype,torch.uint8)
        self.assertEqual(w['0.up.scales'].dtype,torch.uint8)
        self.assertEqual(w['0.up.bias'].dtype,torch.float32)
        self.assertEqual(w['0.router'].dtype,torch.bfloat16)
        self.assertTrue(torch.isfinite(w['0.router']).all())
        self.assertEqual(w['embed'].dtype,torch.float16)

    def test_weight_shape_and_reserved_layout_rejection(self):
        c=validate_config(configuration())
        for name,replacement in [('model.layers.0.mlp.experts.gate_up_proj_blocks',torch.zeros(4,64,32)),
                ('model.layers.0.mlp.experts.down_proj_scales',torch.full((4,32,1),255,dtype=torch.uint8)),
                ('model.layers.0.self_attn.sinks',torch.ones(3)),('unexpected',torch.zeros(1))]:
            state=tensors(c);state[name]=replacement
            with self.subTest(name=name),self.assertRaises(ValueError):load_weights(state,c)


    def test_expert_scale_fast_paths_require_the_entire_tensor_in_range(self):
        from damendaemon.mxfp4 import scale_capabilities
        for low,high,half,normal in ((104,140,True,True),(2,252,False,True),
            (1,140,False,False),(104,253,False,False),(0,0,False,False),
            (127,127,True,True),(103,141,False,True),(254,254,False,False)):
            with self.subTest(low=low,high=high):
                values=torch.full((3,7),127,dtype=torch.uint8);values[0,0]=low;values[-1,-1]=high
                self.assertEqual(scale_capabilities(values),(half,normal))


    def test_long_attention_rejects_tf32_if_global_policy_changes_after_load(self):
        from types import SimpleNamespace
        from damendaemon.gpt_oss import GptOssExecutor
        previous=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            with self.assertRaisesRegex(ValueError,'precision highest'):
                GptOssExecutor.attention_projection(None,SimpleNamespace(shape=(256,32)),None,None,True)
        finally:torch.backends.cuda.matmul.allow_tf32=previous
