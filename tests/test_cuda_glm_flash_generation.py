from pathlib import Path
import tempfile
import unittest
import torch
from unittest.mock import patch
from damendaemon.cuda_glm_flash_generation import CudaFlashGenerator,CudaFlashToolGenerator
from damendaemon.cuda_reference_serving import CudaFlashReferenceRuntime
from damendaemon.reference_serving import ReferenceRuntime
from damendaemon.glm_generation import ReferenceGenerator
from damendaemon.glm_flash_tool_text import FlashToolText
from tests.test_glm_flash_generation import text_fixture,ScriptedFlash


class CudaFlashGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name);base=text_fixture(path)
        self.text=FlashToolText(path,vocab_size=base.vocab_size)
        self.model=ScriptedFlash(self.text,'abc')

    def test_cuda_binding_rejects_cpu_and_does_not_offer_cpu_batch(self):
        with self.assertRaises(ValueError):CudaFlashGenerator(self.model,self.text)
        self.assertFalse(hasattr(CudaFlashToolGenerator,'generate_batch_tools'))
        from damendaemon.glm_flash_generation import FlashReferenceGenerator
        ordinary=FlashReferenceGenerator(self.model,self.text)
        with self.assertRaises(ValueError):CudaFlashReferenceRuntime(ordinary)

    def test_control_generation_cleanup_and_explicit_http_admission(self):
        # CPU control of shared ownership; this does not claim CUDA arithmetic.
        with patch.object(CudaFlashToolGenerator,'model_type',ScriptedFlash),patch.object(CudaFlashToolGenerator,'device_type','cpu'):
            runner=CudaFlashToolGenerator(self.model,self.text,max_context=128)
        with self.assertRaises(ValueError):ReferenceRuntime(runner)
        events=list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=2))
        with self.assertRaises(ValueError):CudaFlashReferenceRuntime(runner)
        runner.device=torch.device('cuda:0') # metadata-only HTTP ownership control
        runtime=CudaFlashReferenceRuntime(runner)
        try:
            self.assertTrue(self.model.states[-1].closed)
            self.assertEqual(events[-1]['finish_reason'],'length')
            with patch.object(runner,'generate',return_value=(e for e in [{'finish_reason':'length','usage':{'prompt_tokens':1,'completion_tokens':1}}])):
                result=list(runtime.generate('model',[dict(role='user',content='Hi')],1))
            self.assertEqual(result[-1]['timing']['source_tier'],'cuda_prepared_reference')
            self.assertEqual(result[-1]['timing']['kernel_backend'],'glm5_next-cuda-prepared-reference')
            status=runtime.manage('status')
            self.assertEqual(status['model_type'],'glm5_next')
            self.assertEqual(status['execution'],'prepared_cuda_flash')
            self.assertEqual(status['scope'],'cuda_prepared_reference')
            self.assertEqual(status['device'],'cuda:0')
            self.assertFalse(status['official_model_qualified'])
            self.assertFalse(status['performance_qualified'])
            self.assertEqual(status['max_batch_size'],1)
            with self.assertRaises(ValueError):runtime.model_manifest()
        finally:runtime.close()
        self.assertEqual(runtime.manage('status')['models'][0]['residency'],'closed')


if __name__=='__main__':unittest.main()
