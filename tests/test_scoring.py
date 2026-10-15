import unittest
from contextlib import contextmanager
from unittest.mock import Mock
from types import SimpleNamespace
import torch
from dllm.scoring import options
from dllm.qwen2 import Executor
from dllm.runtime import Runtime


class ScoringTests(unittest.TestCase):
    def runtime(self, logits):
        runtime = Runtime.__new__(Runtime)
        self.session_released = False
        self.executor = SimpleNamespace(
            model=SimpleNamespace(raw_tokens=lambda _: [1, 2]), context=16,
            score_candidates=lambda *_: logits, reserve_request=Mock(), release_request=Mock(),
            kernel_backend='test-cpu', compute_dtype=torch.float32)
        @contextmanager
        def session(alias):
            try:
                yield self.executor, 'host', 0
            finally:
                self.session_released = True
        runtime.session = session
        return runtime

    def test_wrong_candidate_count_fails_and_releases_owners(self):
        for logits in (torch.empty(0), torch.tensor([1.]), torch.tensor([3., 2., 1.])):
            with self.subTest(logits=logits):
                runtime = self.runtime(logits)
                with self.assertRaisesRegex(RuntimeError, 'candidate score count'):
                    runtime.score('model', 'hello', [3, 7])
                self.executor.reserve_request.assert_called_once_with(0, 2)
                self.executor.release_request.assert_called_once_with(0)
                self.assertTrue(self.session_released)

    def test_valid_candidate_scores_keep_order_and_normalization(self):
        logits = torch.tensor([[1., 3.]])
        result = self.runtime(logits).score('model', 'hello', [7, 3])
        self.assertEqual(result['selected_token_id'], 3)
        self.assertEqual(result['selected_index'], 1)
        self.assertEqual([row['token_id'] for row in result['data']], [7, 3])
        torch.testing.assert_close(torch.tensor([row['probability'] for row in result['data']]),
                                   logits.softmax(-1).reshape(-1))
        self.executor.release_request.assert_called_once_with(0)
        self.assertTrue(self.session_released)

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
        self.assertIsNone(model._candidate_ids)
        self.assertIs(model.w['head'],head)
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            Executor.score_candidates(model,[1],[3,10])
        model.c['model_type']='phi3'
        with self.assertRaisesRegex(ValueError,'unquantized Qwen'):
            Executor.score_candidates(model,[1],[3,7])
