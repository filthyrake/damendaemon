import os
import unittest
from unittest.mock import patch
from dllm.prefill_graphs import configuration


class PrefillGraphTests(unittest.TestCase):
    def test_bounds_and_defaults(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual(configuration(),(0,128))
        for count,tokens in (('-1','128'),('9','128'),('1','0'),('1','513'),('bad','128')):
            with patch.dict(os.environ,{'DLLM_PREFILL_GRAPHS':count,'DLLM_PREFILL_GRAPH_MAX_TOKENS':tokens}):
                with self.assertRaises(ValueError):configuration()

    def test_unqualified_paths_do_not_allocate_cuda_graphs(self):
        from types import SimpleNamespace
        from dllm.qwen2 import Executor
        with patch.dict(os.environ,{'DLLM_PREFILL_GRAPHS':'2'}),patch('dllm.prefill_graphs.PrefillGraphs') as factory:
            for device,family,scales,paged,use_graph in [('mps','qwen2',{},False,True),
                    ('cuda','phi3',{},False,True),('cuda','qwen2',{'scale':1},False,True),
                    ('cuda','qwen2',{},True,True),('cuda','qwen2',{},False,False)]:
                model=SimpleNamespace(device=device,c={'model_type':family},model=SimpleNamespace(scales=scales))
                if paged:model.pool=object()
                Executor.configure_prefill_graphs(model,use_graph)
                self.assertIsNone(model.prefill_graph_cache)
            factory.assert_not_called()
