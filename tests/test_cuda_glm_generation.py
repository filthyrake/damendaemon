from pathlib import Path
import tempfile
import unittest
import torch
from unittest.mock import patch
from dllm.cuda_glm_generation import CudaGlmGenerator,CudaGlmToolGenerator
from dllm.cuda_reference_serving import CudaGlmReferenceRuntime
from dllm.reference_serving import ReferenceRuntime
from dllm.glm_generation import ReferenceGenerator
from dllm.glm_tool_text import GlmToolText
from tests.test_glm_generation import text_fixture,ScriptedModel


class CudaGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name);base=text_fixture(path)
        self.text=GlmToolText(path,vocab_size=base.vocab_size)
        self.model=ScriptedModel(self.text,'abc')

    def test_cuda_binding_rejects_cpu_and_does_not_offer_cpu_batch(self):
        with self.assertRaises(ValueError):CudaGlmGenerator(self.model,self.text)
        self.assertFalse(hasattr(CudaGlmToolGenerator,'generate_batch_tools'))
        ordinary=ReferenceGenerator(self.model,self.text)
        with self.assertRaises(ValueError):CudaGlmReferenceRuntime(ordinary)

    def test_control_generation_cleanup_and_explicit_http_admission(self):
        # CPU control of shared ownership; this does not claim CUDA arithmetic.
        with patch.object(CudaGlmToolGenerator,'model_type',ScriptedModel),patch.object(CudaGlmToolGenerator,'device_type','cpu'):
            runner=CudaGlmToolGenerator(self.model,self.text,max_context=128)
        with self.assertRaises(ValueError):ReferenceRuntime(runner)
        events=list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=2))
        with self.assertRaises(ValueError):CudaGlmReferenceRuntime(runner)
        runner.device=torch.device('cuda:0') # metadata-only HTTP ownership control
        runtime=CudaGlmReferenceRuntime(runner)
        try:
            self.assertTrue(self.model.states[-1].closed)
            self.assertEqual(events[-1]['finish_reason'],'length')
            with patch.object(runner,'generate',return_value=(e for e in [{'finish_reason':'length','usage':{'prompt_tokens':1,'completion_tokens':1}}])):
                result=list(runtime.generate('model',[dict(role='user',content='Hi')],1))
            self.assertEqual(result[-1]['timing']['source_tier'],'cuda_prepared_reference')
            self.assertEqual(result[-1]['timing']['kernel_backend'],'glm_moe_dsa-cuda-prepared-reference')
            status=runtime.manage('status')
            self.assertEqual(status['scope'],'cuda_prepared_reference')
            self.assertEqual(status['device'],'cuda:0')
            self.assertFalse(status['official_model_qualified'])
            self.assertFalse(status['performance_qualified'])
            self.assertEqual(status['max_batch_size'],1)
            with self.assertRaises(ValueError):runtime.model_manifest()
        finally:runtime.close()
        self.assertEqual(runtime.manage('status')['models'][0]['residency'],'closed')


if __name__=='__main__':unittest.main()
