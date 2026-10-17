from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_deepseek_generation import CudaDeepSeekGenerator
from damendaemon.cuda_reference_serving import CudaDeepSeekReferenceRuntime
from damendaemon.reference_serving import ReferenceRuntime
from tests.test_deepseek_generation import text_fixture,ScriptedModel


class CudaDeepSeekGenerationTests(unittest.TestCase):
    def test_explicit_binding_and_shared_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            text=text_fixture(Path(temp));model=ScriptedModel(text,'abc')
            with self.assertRaises(ValueError):CudaDeepSeekGenerator(model,text)
            self.assertFalse(hasattr(CudaDeepSeekGenerator,'generate_batch'))
            with patch.object(CudaDeepSeekGenerator,'model_type',ScriptedModel),patch.object(CudaDeepSeekGenerator,'device_type','cpu'):
                runner=CudaDeepSeekGenerator(model,text,max_context=128)
            events=list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=2))
            self.assertTrue(model.states[-1].closed)
            self.assertEqual(events[-1]['usage']['completion_tokens'],2)
            with self.assertRaises(ValueError):ReferenceRuntime(runner)
            with self.assertRaises(ValueError):CudaDeepSeekReferenceRuntime(runner)
            runner.device=torch.device('cuda:0') # HTTP metadata-only control, not device proof.
            runtime=CudaDeepSeekReferenceRuntime(runner)
            try:
                status=runtime.manage('status')
                self.assertEqual(status['model_type'],'deepseek_v4')
                self.assertEqual(status['execution'],'prepared_cuda_deepseek')
                self.assertFalse(status['performance_qualified'])
                with patch.object(runner,'generate',return_value=(e for e in [{'finish_reason':'length','usage':{'prompt_tokens':1,'completion_tokens':1}}])):
                    final=list(runtime.generate('model',[dict(role='user',content='Hi')],1))[-1]
                self.assertEqual(final['timing']['kernel_backend'],'deepseek_v4-cuda-prepared-reference')
            finally:runtime.close()
