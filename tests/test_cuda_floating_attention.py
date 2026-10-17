import unittest
from unittest.mock import patch
from damendaemon.cuda_floating_attention import attention_prefill,attention_decode
from damendaemon.floating_attention import attention_prefill as cpu_prefill
from tests.test_floating_attention import fixture


class CudaAttentionAdmissionTests(unittest.TestCase):
    def test_cpu_input_rejected_before_core(self):
        args=fixture()
        with patch('damendaemon.cuda_floating_attention.core._attention_prefill',side_effect=AssertionError('entered core')):
            with self.assertRaisesRegex(ValueError,'CUDA'):attention_prefill(*args,groups=2,window=4)

    def test_cpu_state_cannot_enter_cuda_decode(self):
        hidden,weights,freq=fixture()
        _,state=cpu_prefill(hidden[:,:3],weights,freq,groups=2,window=4,max_tokens=13)
        try:
            with self.assertRaisesRegex(ValueError,'CUDA state'):attention_decode(hidden[:,3:4],state,position=3)
        finally:state.close()

    def test_missing_inputs_and_limits_rejected(self):
        with self.assertRaises(ValueError):attention_prefill(None,None,None,groups=2,window=4)
        with self.assertRaises(ValueError):attention_prefill(*fixture(),groups=2,window=4,max_workspace_bytes=True)
