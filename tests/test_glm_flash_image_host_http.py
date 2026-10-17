from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from damendaemon.glm_flash_host_image_serving import HostFlashImageReferenceRuntime,QueuedHostFlashImageReferenceRuntime
from damendaemon.glm_flash_host_serving import HostFlashReferenceRuntime
from damendaemon.reference_serving import ReferenceRuntime
from damendaemon.runtime import BusyError
from damendaemon.glm_flash_tools import FlashTools
from damendaemon.glm_text import END_THINK,EOS
from tests import test_glm_flash_host_image_generation as native,test_glm_flash_host_serving as http
from tests.test_glm_flash_inline_images import messages as inline_messages
from tests.test_glm_tools import TOOLS,CALLS


class FlashImageHostHttpTests(unittest.TestCase):
    setUp=native.FlashHostImageGenerationTests.setUp
    session=native.FlashHostImageGenerationTests.session
    pump=native.FlashHostImageGenerationTests.pump
    scripted=native.FlashHostImageGenerationTests.scripted
    control=native.FlashHostImageGenerationTests.control
    serve=http.FlashHostServingTests.serve

    def runtime(self,queued=False,**options):
        runner,owners,_=self.session(verified=True)
        kind=QueuedHostFlashImageReferenceRuntime if queued else HostFlashImageReferenceRuntime
        runtime=kind(runner,**options);self.addCleanup(runtime.close)
        return runtime,owners

    def post(self,**options):
        return http.FlashHostServingTests.post(self,**(dict(messages=inline_messages())|options))

    def until(self,condition):
        end=time.monotonic()+4
        while not condition() and time.monotonic()<end:threading.Event().wait(.01)
        self.assertTrue(condition())

    def test_authenticated_json_sse_full_control_modality_and_verified_status(self):
        from damendaemon.glm_flash_inline_images import decode_inline_images
        runtime,owners=self.runtime();base=self.serve(runtime);decoded=decode_inline_images(inline_messages())
        expected=list(self.control().generate_images(decoded.messages,decoded.images,max_new_tokens=3))
        with self.assertRaises(HTTPError) as error:urlopen(base+'/v1/models/status',timeout=3)
        self.assertEqual(error.exception.code,401);error.exception.close()
        with self.post() as response:actual=json.load(response)
        message=actual['choices'][0]['message']
        for key in ('content','reasoning_content'):self.assertEqual(message.get(key) or '', ''.join(event.get(key,'') for event in expected))
        self.assertEqual(actual['usage']['prompt_tokens'],expected[-1]['usage']['prompt_tokens'])
        self.assertTrue(actual['damendaemon_timing']['vision_identity_verified'])
        self.assertEqual(actual['damendaemon_timing']['vision_device'],'cpu')
        with self.post(stream=True,stream_options={'include_usage':True}) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire);self.assertIn('host_topology',wire)
        with urlopen(Request(base+'/v1/models/status',headers=self.admin),timeout=3) as response:status=json.load(response)
        self.assertEqual(status['vision_device'],'cpu')
        self.assertEqual(status['modality'],'still_image');self.assertEqual(status['stage_layers'],[[0,2],[2,3]])
        self.assertTrue(status['model_identity_verified']);self.assertTrue(status['vision_identity_verified'])
        self.assertFalse(status['official_model_qualified']);self.assertFalse(status['epoch_terminal'])
        self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))

    def test_native_image_tool_json_sse_and_output_exclusion(self):
        from damendaemon.glm_flash_inline_images import decode_inline_images
        runtime,owners=self.runtime();self.serve(runtime);decoded=decode_inline_images(inline_messages(),tools=TOOLS)
        self.prompt_length=len(self.text.image_tool_chat_tokens(decoded.messages,[[1,8,8]],TOOLS))
        wire='reason'+END_THINK+FlashTools(TOOLS).render(CALLS)+EOS
        for streaming in (False,True):
            with self.scripted(runtime.runner,owners,wire),self.post(tools=TOOLS,max_tokens=512,stream=streaming) as response:
                if streaming:
                    result=response.read().decode();self.assertIn('data: [DONE]',result);self.assertIn('tool_calls',result)
                else:
                    result=json.load(response);choice=result['choices'][0]
                    self.assertEqual(choice['finish_reason'],'tool_calls')
                    call=choice['message']['tool_calls'][0]
                    self.assertEqual(json.loads(call['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        self.assertTrue(runtime.is_healthy())

    def test_exact_image_admission_and_invalid_http_preserve_epoch(self):
        runtime,owners=self.runtime();self.serve(runtime)
        for kind in (ReferenceRuntime,HostFlashReferenceRuntime,HostFlashImageReferenceRuntime):
            with self.assertRaises(ValueError):kind(runtime.runner)
        with patch.object(owners[0],'prefill_embeddings',side_effect=AssertionError('invalid dispatch')):
            for options in (dict(messages=[]),dict(messages=[dict(role='user',content='text only')]),dict(temperature=1),
                    dict(max_tokens=4096),dict(messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':'https://invalid.test/image'})])]),
                    dict(tools=TOOLS,tool_choice='required')):
                with self.assertRaises(HTTPError) as error:self.post(**options)
                self.assertEqual(error.exception.code,400);error.exception.close()
        self.assertTrue(runtime.is_healthy())
        with self.post() as response:self.assertEqual(response.status,200)

    def test_active_abandonment_peer_failure_and_retained_cleanup(self):
        for mode in ('abandon','peer','cleanup'):
            runtime,owners=self.runtime();stream=runtime.generate('model',inline_messages(),8);next(stream)
            if mode=='peer':
                owners[-1].close()
                with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):list(stream)
            elif mode=='cleanup':
                with patch.object(owners[0].stage,'close',side_effect=RuntimeError('retry stage')):
                    stream.close();self.assertTrue(runtime.cleanup_pending());self.assertIsNotNone(runtime.runner)
                runtime.close();self.assertIsNone(runtime.runner)
            else:stream.close()
            self.assertFalse(runtime.is_healthy());self.assertTrue(runtime.closed)

    def test_queue_concurrent_http_controls_tools_and_authenticated_drain(self):
        runtime,owners=self.runtime(queued=True,queue_size=4);self.serve(runtime)
        bad=inline_messages();bad[0]['content'][1]['image_url']['url']='data:image/png;base64,aGVsbG8='
        with self.assertRaises(HTTPError) as error:self.post(messages=bad)
        self.assertEqual(error.exception.code,400);error.exception.close();self.assertTrue(runtime.is_healthy())
        def request():
            with self.post() as response:return json.load(response)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda _:request(),range(3)))
        self.assertTrue(all(result['choices']==results[0]['choices'] for result in results))
        self.assertTrue(all(result['damendaemon_timing']['queue_seconds']>=0 for result in results))
        with self.post(tools=TOOLS,max_tokens=1) as response:self.assertEqual(json.load(response)['usage']['completion_tokens'],1)
        self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))
        with urlopen(Request(self.base+'/v1/scheduler/drain',b'{}',self.admin),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
        with self.assertRaises(HTTPError) as error:self.post()
        self.assertEqual(error.exception.code,503);error.exception.close();self.assertTrue(runtime.is_healthy())

    def test_queue_overflow_expiry_and_snapshot_before_pixels(self):
        runtime,owners=self.runtime(queued=True,queue_size=1,queue_timeout=.2)
        entered=threading.Event();release=threading.Event();original=owners[-1].stage.forward_layers
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(4):raise RuntimeError('test timeout')
            return original(*args,**kwargs)
        with patch.object(owners[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)));self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            try:
                with self.assertRaisesRegex(BusyError,'full'):list(runtime.generate('model',inline_messages(),1))
                with self.assertRaisesRegex(BusyError,'deadline'):second.result(2)
            finally:release.set()
            self.assertEqual(first.result(3)[-1]['finish_reason'],'length')
        self.until(lambda:not runtime.accepted);self.assertTrue(runtime.is_healthy())
        runtime,owners=self.runtime(queued=True,queue_size=2);entered.clear();release.clear();original=owners[-1].stage.forward_layers
        source=inline_messages()
        with patch.object(owners[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)));self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',source,1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            source[0]['content'][1]['image_url']['url']='invalid'
            release.set();self.assertEqual(first.result(3)[-1]['finish_reason'],'length')
            self.assertEqual(second.result(3)[-1]['finish_reason'],'length')
        self.assertTrue(runtime.is_healthy())

    def test_queue_active_abandonment_flushes_waiters_and_backpressure_pings(self):
        runtime,owners=self.runtime(queued=True,queue_size=2)
        stream=runtime.generate('model',inline_messages(),128);next(stream)
        self.until(lambda:next(iter(runtime.accepted.values())).events.qsize()>=8)
        # A full event queue may still have one decode already in flight.
        # Observe PING at a command boundary, under its own worker lock.
        with owners[0].lock:
            before=list(owners[0].channel.state.lengths);owners[0].ping()
            self.assertEqual(owners[0].channel.state.lengths,before)
        with ThreadPoolExecutor(1) as pool:
            waiting=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            stream.close()
            with self.assertRaises(BusyError):waiting.result(3)
        self.until(lambda:not runtime.worker.is_alive());self.assertFalse(runtime.accepted);self.assertFalse(runtime.is_healthy())

    def test_queue_close_interrupts_receive_and_flushes_waiters(self):
        runtime,owners=self.runtime(queued=True,queue_size=2);entered=threading.Event();release=threading.Event()
        original=owners[-1].stage.forward_layers
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(4):raise RuntimeError('test timeout')
            return original(*args,**kwargs)
        with patch.object(owners[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(2) as pool:
            first=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)));self.assertTrue(entered.wait(2))
            second=pool.submit(lambda:list(runtime.generate('model',inline_messages(),1)))
            self.until(lambda:runtime.scheduler_status()['queued_generation_requests']==1)
            try:
                runtime.close()
                for future in (first,second):
                    with self.assertRaises((BusyError,RuntimeError,ValueError,EOFError,OSError)):future.result(2)
            finally:release.set()
        self.assertTrue(runtime.closed);self.assertFalse(runtime.worker.is_alive());self.assertFalse(runtime.accepted)
