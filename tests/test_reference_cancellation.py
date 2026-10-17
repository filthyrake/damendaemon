from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from damendaemon.reference_cancellation import cancellation_scope,check_cancelled,ReferenceCancelled
from damendaemon import deepseek_generation
from damendaemon.runtime import BusyError
from tests import test_queued_reference_serving as queues
from tests import test_reference_serving as fixtures


class ReferenceCancellationTests(unittest.TestCase):
    def test_scopes_combine_restore_and_do_not_leak_to_threads(self):
        outer=threading.Event();inner=threading.Event()
        with cancellation_scope(outer.is_set):
            check_cancelled()
            with cancellation_scope(inner.is_set):
                inner.set()
                with self.assertRaises(ReferenceCancelled):check_cancelled()
            check_cancelled();outer.set()
            with ThreadPoolExecutor(1) as pool:pool.submit(check_cancelled).result(2)
            with self.assertRaises(ReferenceCancelled):check_cancelled()
        check_cancelled()
        with self.assertRaises(ValueError):
            with cancellation_scope(None):pass

    def test_queued_abandonment_interrupts_buffered_calls_before_completion(self):
        fixture=queues.QueuedReferenceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        for family in ('glm','flash','deepseek'):
            runtime,model=fixture.runtime(family);entered=threading.Event();resume=threading.Event()
            decode_name='decode' if family=='deepseek' else 'forward';decode=getattr(model,decode_name)
            boundary=len(model.tokens)//2;streams=[]
            original=deepseek_generation.TokenStream if family=='deepseek' else runtime.runner.stream_type
            def retained(*args,**kwargs):
                stream=original(*args,**kwargs);streams.append(stream);return stream
            target=deepseek_generation if family=='deepseek' else runtime.runner
            attribute='TokenStream' if family=='deepseek' else 'stream_type'
            def blocked_decode(*args,**kwargs):
                logits=decode(*args,**kwargs)
                if model.index==boundary:
                    entered.set();self.assertTrue(resume.wait(3))
                return logits
            options=dict(enable_thinking=True) if family=='deepseek' else {}
            with patch.object(target,attribute,side_effect=retained),patch.object(model,decode_name,side_effect=blocked_decode):
                generation=fixture.generate(runtime,**options)
                first=next(generation);self.assertNotIn('tool_calls',first)
                deadline=time.monotonic()+3
                request=next(iter(runtime.accepted.values()))
                while not entered.is_set() and time.monotonic()<deadline:
                    if request.events.qsize():self.assertNotIn('tool_calls',next(generation))
                    else:entered.wait(.001)
                self.assertTrue(entered.is_set())
                parser=streams[0].parser
                buffer=parser.tool_buffer if family=='deepseek' else parser.buffer
                self.assertIsNotNone(buffer);self.assertFalse(buffer.closed)
                generation.close();resume.set()
                queues.wait_for(lambda:not runtime.accepted)
                self.assertEqual(model.index,boundary)
                self.assertLess(model.index,len(model.tokens)-1)
                self.assertTrue(model.states[0].closed);self.assertTrue(buffer.closed)
                self.assertFalse(runtime.stopping.is_set())
            # The cancellation scope is gone; a fresh request still returns its call.
            events=list(fixture.generate(runtime,**options))
            self.assertEqual(events[-1]['finish_reason'],'tool_calls')
            self.assertTrue(model.states[-1].closed);runtime.close()

    def test_runtime_shutdown_cancels_before_prefill_and_preserves_failed_release(self):
        fixture=fixtures.ReferenceServingTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        for family in ('glm','flash','deepseek'):
            runtime,model=fixture.runtime(family,tools=True)
            options=dict(enable_thinking=True) if family=='deepseek' else {}
            with cancellation_scope(lambda:True),patch.object(model,'prefill',side_effect=AssertionError('prefill')):
                with self.assertRaises(BusyError):list(runtime.generate('model',fixture.messages,512,tools=fixtures.TOOLS,**options))
            self.assertFalse(model.states)
            decode_name='decode' if family=='deepseek' else 'forward';decode=getattr(model,decode_name)
            cancelled=threading.Event();observed=[]
            def cancel(*args,**kwargs):
                result=decode(*args,**kwargs)
                if model.index==len(model.tokens)//2:cancelled.set()
                return result
            with cancellation_scope(cancelled.is_set),patch.object(model,decode_name,side_effect=cancel):
                generation=runtime.generate('model',fixture.messages,512,tools=fixtures.TOOLS,**options)
                observed.append(next(generation));state=model.states[-1]
                with patch.object(state,'close',side_effect=RuntimeError('release failed')):
                    with self.assertRaises(BusyError) as error:
                        for event in generation:observed.append(event)
                    self.assertIsInstance(error.exception.__cause__,ReferenceCancelled)
                    self.assertTrue(error.exception.__cause__.__notes__)
                    self.assertTrue(runtime.cleanup_pending())
                runtime.runner.cleanup()
            self.assertTrue(state.closed);self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in observed))
            runtime.close()
