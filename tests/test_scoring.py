import unittest
from types import SimpleNamespace
import torch
from dllm.scoring import options
from dllm.qwen2 import Executor


class ScoringTests(unittest.TestCase):
    def test_bounds_and_copy(self):
        prompt=[1,2];ids=[3,4]
        copied,selected=options(prompt,ids);prompt.append(5);ids.append(6)
        self.assertEqual(copied,[1,2]);self.assertEqual(selected,[3,4])
        for value in ([],[1],[1,1],[True,2],[-1,2],[1,2.0],list(range(257)),None):
            with self.subTest(value=value),self.assertRaises(ValueError):options('test',value)

    def test_head_override_is_removed_after_failed_prefill(self):
        head=torch.arange(40,dtype=torch.float32).view(10,4)
        model=SimpleNamespace(c={'model_type':'qwen2'},w={'head':head},device='cpu')
        def fail(tokens):
            torch.testing.assert_close(model._candidate_head,head[[3,7]])
            raise ValueError('prefill failure')
        model.prefill=fail
        with self.assertRaisesRegex(ValueError,'prefill failure'):
            Executor.score_candidates(model,[1,2],[3,7])
        self.assertIsNone(model._candidate_head)
        self.assertIs(model.w['head'],head)
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            Executor.score_candidates(model,[1],[3,10])
        model.c['model_type']='phi3'
        with self.assertRaisesRegex(ValueError,'unquantized Qwen'):
            Executor.score_candidates(model,[1],[3,7])
