import os
import unittest
from unittest.mock import patch

import torch
from dllm.native_q4_residual import library,linear,qualified_scope,select
from dllm.quantization import Linear


def config():
    return dict(model_type='llama',hidden_size=4096,intermediate_size=14336,num_hidden_layers=32,
                num_attention_heads=32,num_key_value_heads=8,vocab_size=32000,head_dim=128)


class NativeQ4ResidualTests(unittest.TestCase):
    def tearDown(self):library.cache_clear()

    def test_scope_rejects_unqualified_precision_context_and_topology(self):
        def allowed(c=None,quantization='q4_0',device='cuda',dtype=torch.float16,precision='mixed',context=512):
            return qualified_scope(c or config(),quantization,device,dtype,precision,context)
        self.assertTrue(allowed())
        for kwargs in [dict(quantization='q8_0'),dict(device='mps'),dict(dtype=torch.float32),
                       dict(precision='standard'),dict(context=513)]:
            self.assertFalse(allowed(**kwargs))
        for key,value in [('model_type','mistral'),('num_key_value_heads',32),('vocab_size',128256),
                          ('head_dim',64),('num_hidden_layers',16),('hidden_size',2048)]:
            self.assertFalse(allowed({**config(),key:value}))

    def test_selected_path_is_bound_to_executor_and_batch_falls_back(self):
        args=(config(),'q4_0','cuda',torch.float16,'mixed',512)
        with patch.dict(os.environ,{'DLLM_Q4_RESIDUAL_LIBRARY':''}),patch('ctypes.CDLL') as load:
            self.assertIsNone(select(*args));load.assert_not_called()
        with patch.dict(os.environ,{'DLLM_Q4_RESIDUAL_LIBRARY':'first'}),patch('dllm.native_q4_residual.library'):
            decode=select(*args)
        x=torch.zeros(1,4096,dtype=torch.float16);w=torch.empty(1024,2048,dtype=torch.uint8)
        scales=torch.empty(1024,128,dtype=torch.float16)
        with patch.dict(os.environ,{'DLLM_Q4_RESIDUAL_LIBRARY':'second'}),patch('dllm.native_q4_residual.linear',return_value='selected') as multiply:
            self.assertIsNone(decode(x.expand(2,-1),w,scales));multiply.assert_not_called()
            self.assertEqual(decode(x,w,scales),'selected')
            self.assertEqual(multiply.call_args.kwargs['library_path'],'first')

    def test_abi_and_cpu_buffers_fail_before_launch(self):
        handle=type('Library',(),{})();handle.dllm_q4_residual_abi_version=lambda:1
        with patch('ctypes.CDLL',return_value=handle),self.assertRaisesRegex(RuntimeError,'ABI'):
            library('/old-experiment')
        with patch('ctypes.CDLL') as load,self.assertRaisesRegex(ValueError,'buffers'):
            linear(torch.zeros(1,256,dtype=torch.float16),torch.zeros(2,128,dtype=torch.uint8),
                   torch.ones(2,8,dtype=torch.float16),library_path='/not-loaded')
        load.assert_not_called()

    def test_per_model_q4_dispatch_leaves_dense_weights_on_fallback(self):
        quant=torch.zeros(2,16,dtype=torch.uint8);dense=torch.ones(2,32);scales=torch.ones(2,1,dtype=torch.float16)
        selected=object();fallback=object()
        call=Linear({'q':quant},{'q':scales},'cpu',lambda *args:fallback,q4_decode=lambda *args:selected)
        self.assertIs(call(torch.ones(1,32),quant),selected)
        self.assertIs(call(torch.ones(1,32),dense),fallback)


if __name__=='__main__':unittest.main()
