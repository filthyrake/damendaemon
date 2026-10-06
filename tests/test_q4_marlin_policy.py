import unittest

import torch

from dllm.q4_marlin_policy import plan


class Q4MarlinPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(model_type='llama',hidden_size=4096,intermediate_size=14336,
            num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,vocab_size=32000,head_dim=128)
        shapes={'head':(32000,4096)}
        for layer in range(32):
            shapes.update({f'{layer}.{name}':shape for name,shape in
                {'qkv.weight':(6144,4096),'out':(4096,4096),
                 'gate_up':(28672,4096),'down':(4096,14336)}.items()})
        self.weights={name:torch.empty((n,k//2),dtype=torch.uint8,device='meta') for name,(n,k) in shapes.items()}
        self.scales={name:torch.empty((n,k//32),dtype=torch.float16,device='meta') for name,(n,k) in shapes.items()}
        self.weights['embed']=torch.empty((32000,4096),dtype=torch.float16,device='meta')
        self.args=dict(config=self.config,weights=self.weights,scales=self.scales,quantization='q4_0',
            device='cuda',dtype=torch.float16,precision='mixed',context=512,max_batch=4,fused=True,device_name='NVIDIA A2')

    def test_plan_accounts_for_replacement_overhead_and_conversion_peak(self):
        selected=plan(**self.args)
        self.assertEqual(len(selected.names),129)
        self.assertEqual(selected.canonical_bytes,3999596544)
        self.assertEqual(selected.prepared_bytes,4110696448)
        self.assertEqual(selected.overhead_bytes,111099904)
        self.assertEqual(selected.repack_peak_bytes,81920512)
        self.assertLess(selected.overhead_bytes+selected.repack_peak_bytes,selected.canonical_bytes//20)

    def test_unqualified_precision_hardware_context_and_geometry_fall_back(self):
        changes=({'device':'mps'},{'device_name':'NVIDIA A30'},{'dtype':torch.float32},
            {'precision':'float32'},{'context':0},{'context':513},{'max_batch':0},
            {'max_batch':5},{'fused':False},{'quantization':'q8_0'},
            {'config':{**self.config,'model_type':'mistral'}},
            {'config':{**self.config,'num_key_value_heads':32}})
        for change in changes:
            with self.subTest(change=change):self.assertFalse(plan(**{**self.args,**change}).names)

    def test_aliases_biases_and_partial_layouts_are_not_replaced(self):
        cases=[dict(weights={**self.weights,'embed':self.weights['head']}),
               dict(weights={**self.weights,'0.out':self.weights['1.out']}),
               dict(scales={**self.scales,'0.out':self.scales['1.out']}),
               dict(weights={key:value for key,value in self.weights.items() if key!='head'}),
               dict(weights={**self.weights,'0.qkv.bias':torch.empty(6144,device='meta')}),
               dict(scales={**self.scales,'head':torch.empty((32000,128),dtype=torch.float32,device='meta')}),
               dict(weights={**self.weights,'0.down':torch.empty((4096,4096),dtype=torch.uint8,device='meta')})]
        for change in cases:
            with self.subTest(fields=list(change)):self.assertFalse(plan(**{**self.args,**change}).names)


if __name__=='__main__':unittest.main()
