"""Direct native iterator success follows KV and model lease cleanup."""
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from damendaemon.runtime import Runtime


class NativeCompletionTests(unittest.TestCase):
    def test_terminal_follows_both_cleanup_boundaries(self):
        for failure in (None,'kv','session'):
            with self.subTest(failure=failure):
                state=dict(kv=False,session=False)
                model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':3},eos=set(),quantization='none')
                class Executor:
                    kernel_backend='fake'
                    residual_dtype='torch.float32'
                    def reserve_request(self,*args):state['kv']=True
                    def release_request(self,*args):
                        state['kv']=False
                        if failure=='kv':raise RuntimeError('KV release failed')
                    def prefill(self,prompt):return torch.tensor([[0.,1.,2.]])
                executor=Executor();executor.model=model
                runtime=Runtime.__new__(Runtime)
                runtime.context=1024
                @contextmanager
                def session(alias):
                    state['session']=True
                    try:yield executor,'disk',0
                    finally:
                        state['session']=False
                        if failure=='session':raise RuntimeError('session release failed')
                runtime.session=session
                with ExitStack() as stack:
                    stack.enter_context(patch('damendaemon.harmony.output_stream',return_value=None))
                    stack.enter_context(patch('damendaemon.prompts.encode',return_value=[1]))
                    stack.enter_context(patch('damendaemon.prompts.validate_lengths'))
                    stack.enter_context(patch('damendaemon.runtime.decode_output',return_value='ok'))
                    stream=runtime.generate('a',[],1)
                    try:
                        self.assertEqual(next(stream)['text'],'ok')
                        self.assertTrue(state['kv']);self.assertTrue(state['session'])
                        if failure:
                            with self.assertRaisesRegex(RuntimeError,'release failed'):next(stream)
                        else:
                            terminal=next(stream)
                            self.assertEqual(terminal['finish_reason'],'length')
                            self.assertFalse(state['kv']);self.assertFalse(state['session'])
                            self.assertEqual(terminal['usage']['completion_tokens'],1)
                            with self.assertRaises(StopIteration):next(stream)
                    finally:stream.close()
                self.assertFalse(state['kv']);self.assertFalse(state['session'])

    def test_nonfinite_greedy_prefill_releases_before_any_output(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            state=dict(kv=False,session=False)
            model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':3},eos=set(),quantization='none')
            class Executor:
                def reserve_request(self,*args):state['kv']=True
                def release_request(self,*args):state['kv']=False
                def prefill(self,prompt):return torch.full((1,3),value)
            executor=Executor();executor.model=model
            runtime=Runtime.__new__(Runtime);runtime.context=16
            @contextmanager
            def session(alias):
                state['session']=True
                try:yield executor,'host',0
                finally:state['session']=False
            runtime.session=session
            with patch('damendaemon.harmony.output_stream',return_value=None), \
                    patch('damendaemon.prompts.encode',return_value=[1]), \
                    patch('damendaemon.prompts.validate_lengths'):
                with self.subTest(value=value),self.assertRaisesRegex(RuntimeError,'non-finite greedy'):
                    next(runtime.generate('a',[],1))
            self.assertFalse(state['kv']);self.assertFalse(state['session'])
