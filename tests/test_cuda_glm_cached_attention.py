import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_glm_cached_attention import CudaCachedGlmAttention,CudaGlmIndexer
from damendaemon.glm_cached_attention import CachedGlmAttention
from tests.test_glm_cached_attention import fixture


class CudaGlmCacheTests(unittest.TestCase):
    def test_cuda_wrappers_reject_cpu_weights(self):
        model,_=fixture()
        with self.assertRaises(ValueError):CudaCachedGlmAttention(model.weights)
        with self.assertRaises(ValueError):CudaGlmIndexer(model.indexer.weights,rope_dim=4)

    def test_fake_device_cache_ownership_budget_and_close(self):
        model,_=fixture()
        from torch._subclasses.fake_tensor import FakeTensorMode
        options=dict(heads=2,nope_dim=2,rope_dim=4,value_dim=3,index_top_k=3)
        with FakeTensorMode(),patch('damendaemon.glm_cached_attention._finite_parameter',return_value=True),patch('damendaemon.glm_indexer._finite_parameter',return_value=True):
            weights={name:torch.empty(value.shape,dtype=value.dtype,device='cuda:0') for name,value in model.weights.items()}
            prepared=CudaCachedGlmAttention(weights,**options)
            with self.assertRaises(ValueError):CachedGlmAttention(weights,**options)
            with self.assertRaises(ValueError):prepared.new_state(2,7,max_cache_bytes=1)
            state=prepared.new_state(2,7)
            self.assertEqual(state.latents.device,torch.device('cuda:0'))
            with self.assertRaises(ValueError):state.reorder(torch.zeros(1,dtype=torch.int64))
            other=CudaCachedGlmAttention(weights,**options)
            with self.assertRaises(ValueError):other.forward(torch.ones(2,1,6,device='cuda:0'),state,position=0)
            with self.assertRaises(ValueError):prepared.forward(torch.ones(2,1,6,device='cuda:1'),state,position=0)
            state.close();state.close()
            self.assertIsNone(state.latents);self.assertIsNone(state.owner)
            with self.assertRaises(RuntimeError):prepared.forward(torch.ones(2,1,6,device='cuda:0'),state,position=0)


if __name__=='__main__':unittest.main()
