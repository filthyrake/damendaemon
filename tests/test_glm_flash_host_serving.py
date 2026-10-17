import json
import secrets
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from damendaemon.glm_flash_host_generation import HostFlashGenerator
from damendaemon.glm_flash_host_serving import HostFlashReferenceRuntime
from damendaemon.reference_serving import ReferenceRuntime,handler_for_reference
from damendaemon.runtime import BusyError
from damendaemon.server import BoundedHTTPServer
from damendaemon.http_auth import AuthPolicy
from tests import test_glm_flash_host_generation as native
from tests.test_glm_tools import TOOLS,CALLS


class FlashHostServingTests(unittest.TestCase):
    session=native.FlashHostGenerationTests.session
    links=native.FlashHostGenerationTests.links
    together=native.FlashHostGenerationTests.together
    pump=native.FlashHostGenerationTests.pump
    scripted=native.FlashHostGenerationTests.scripted

    def runtime(self):
        text,workers,control=self.session();runner=HostFlashGenerator(workers[0],text,chunk_size=2048)
        runtime=HostFlashReferenceRuntime(runner);self.addCleanup(runtime.close)
        return runtime,text,workers,control

    def serve(self,runtime):
        key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
        with patch.dict('os.environ',{'DAMENDAEMON_API_KEY':key,'DAMENDAEMON_ADMIN_API_KEY':admin},clear=True):auth=AuthPolicy.from_env()
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for_reference(runtime,auth))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def close():server.shutdown();server.server_close();thread.join(3)
        self.addCleanup(close)
        self.base=f'http://127.0.0.1:{server.server_port}'
        self.headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
        self.admin=self.headers|{'Authorization':'Bearer '+admin}
        return self.base

    def post(self,**options):
        body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=3)|options
        return urlopen(Request(self.base+'/v1/chat/completions',json.dumps(body).encode(),self.headers),timeout=6)

    def test_authenticated_actual_host_json_sse_control_and_status(self):
        runtime,text,workers,control=self.runtime();base=self.serve(runtime)
        with self.assertRaises(HTTPError) as error:urlopen(base+'/v1/models/status',timeout=3)
        self.assertEqual(error.exception.code,401);error.exception.close()
        expected=list(control.generate([dict(role='user',content='Hi')],max_new_tokens=3))
        with self.post() as response:actual=json.load(response)
        message=actual['choices'][0]['message']
        for key in ('content','reasoning_content'):
            self.assertEqual(message.get(key) or '', ''.join(e.get(key,'') for e in expected))
        self.assertEqual(actual['usage']['completion_tokens'],expected[-1]['usage']['completion_tokens'])
        self.assertEqual(actual['damendaemon_timing']['execution'],'host_topology')
        with self.post(stream=True,stream_options={'include_usage':True}) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire);self.assertIn('host_topology',wire)
        with urlopen(Request(base+'/v1/models/status',headers=self.admin),timeout=3) as response:status=json.load(response)
        self.assertEqual(status['model_type'],'glm5_next');self.assertEqual(status['stage_layers'],[[0,2],[2,3]]);self.assertFalse(status['model_identity_verified'])
        self.assertFalse(status['official_model_qualified']);self.assertFalse(status['epoch_terminal'])
        self.assertEqual(status['source_model_id'],workers[0].topology.source_identity.hex())
        self.assertTrue(all(worker.stage.states==[None,None] for worker in workers))
        with urlopen(Request(base+'/health',headers=self.headers),timeout=3) as response:self.assertEqual(response.status,200)

    def test_native_tool_json_sse_over_real_host_execution(self):
        runtime,text,workers,_=self.runtime();self.serve(runtime)
        for streaming in (False,True):
            tokens,step,selection=self.scripted(workers[-1],text)
            with selection,self.post(tools=TOOLS,max_tokens=len(tokens),stream=streaming) as response:
                if streaming:
                    wire=response.read().decode();self.assertIn('data: [DONE]',wire)
                    self.assertIn('tool_calls',wire);self.assertIn(CALLS[0]['function']['name'],wire)
                else:
                    result=json.load(response);choice=result['choices'][0]
                    self.assertEqual(choice['finish_reason'],'tool_calls')
                    call=choice['message']['tool_calls'][0]
                    self.assertEqual(call['function']['name'],CALLS[0]['function']['name'])
                    self.assertEqual(json.loads(call['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        self.assertTrue(runtime.is_healthy())

    def test_explicit_admission_and_invalid_http_requests_preserve_session(self):
        runtime,text,workers,_=self.runtime();runner=runtime.runner;self.serve(runtime)
        with self.assertRaises(ValueError):ReferenceRuntime(runner)
        from damendaemon.glm_host_serving import HostGlmReferenceRuntime
        with self.assertRaises(ValueError):HostGlmReferenceRuntime(runner)
        with self.assertRaises(ValueError):HostFlashReferenceRuntime(runner)
        with patch.object(workers[0],'prefill_chunk',side_effect=AssertionError('dispatch')):
            for options in (dict(temperature=1),dict(max_tokens=4096),dict(messages=[]),dict(messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':'https://invalid.test/image'})])]),dict(tool_choice='required',tools=TOOLS)):
                with self.assertRaises(HTTPError) as error:self.post(**options)
                self.assertEqual(error.exception.code,400);error.exception.close()
        self.assertTrue(runtime.is_healthy())
        with self.post() as response:self.assertEqual(response.status,200)

    def test_abandonment_busy_close_and_remote_failure_are_terminal(self):
        messages=[dict(role='user',content='Hi')]
        for mode in ('abandon','close','remote'):
            runtime,_,workers,_=self.runtime();stream=runtime.generate('model',messages,max_tokens=8);next(stream)
            with self.assertRaises(BusyError):next(runtime.generate('model',messages,max_tokens=1))
            if mode=='abandon':stream.close()
            elif mode=='close':
                with self.assertRaises(BusyError):runtime.close()
                with self.assertRaises((BusyError,RuntimeError)):list(stream)
            else:
                workers[-1].close()
                with self.assertRaises((RuntimeError,EOFError,OSError,ValueError)):list(stream)
            self.assertFalse(runtime.is_healthy());self.assertTrue(runtime.closed)
            with self.assertRaises(BusyError):list(runtime.generate('model',messages,max_tokens=1))

    def test_retained_cleanup_failure_is_unhealthy_and_retryable(self):
        runtime,_,workers,_=self.runtime();base=self.serve(runtime);runner=runtime.runner
        stream=runtime.generate('model',[dict(role='user',content='Hi')],max_tokens=8);next(stream)
        with patch.object(workers[0].stage,'close',side_effect=RuntimeError('stage cleanup')):
            stream.close()
            self.assertFalse(runtime.closed);self.assertIs(runtime.runner,runner)
            self.assertTrue(runtime.cleanup_pending());self.assertFalse(runtime.is_healthy())
            with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/health',headers=self.headers),timeout=3)
            self.assertEqual(error.exception.code,503);error.exception.close()
            with self.assertRaisesRegex(RuntimeError,'stage cleanup'):runtime.close()
        runtime.close();self.assertTrue(runtime.closed);self.assertIsNone(runtime.runner)
        self.assertFalse(runtime.cleanup_pending())

    def test_close_interrupts_blocked_host_receive(self):
        from concurrent.futures import ThreadPoolExecutor
        runtime,_,workers,_=self.runtime();entered=threading.Event();release=threading.Event()
        original=workers[-1].stage.forward_layers
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise RuntimeError('test release timeout')
            return original(*args,**kwargs)
        with patch.object(workers[-1].stage,'forward_layers',blocked),ThreadPoolExecutor(1) as pool:
            future=pool.submit(lambda:list(runtime.generate('model',[dict(role='user',content='Hi')],max_tokens=2)))
            try:
                self.assertTrue(entered.wait(3))
                try:runtime.close()
                except BusyError:pass
                with self.assertRaises((RuntimeError,EOFError,OSError,ValueError)):future.result(2)
            finally:release.set()
        runtime.close();self.assertTrue(runtime.closed);self.assertFalse(runtime.is_healthy())

    def test_http_remote_failure_has_no_sse_done_and_readiness_fails(self):
        runtime,_,workers,_=self.runtime();base=self.serve(runtime)
        original=workers[-1].stage.forward_layers;count=0
        def fail_later(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:raise RuntimeError('remote decode failure')
            return original(*args,**kwargs)
        with patch.object(workers[-1].stage,'forward_layers',fail_later):
            with self.post(stream=True,max_tokens=8) as response:wire=response.read().decode()
        self.assertNotIn('data: [DONE]',wire)
        self.assertFalse(runtime.is_healthy());self.assertTrue(runtime.closed)
        with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/health',headers=self.headers),timeout=3)
        self.assertEqual(error.exception.code,503);error.exception.close()
