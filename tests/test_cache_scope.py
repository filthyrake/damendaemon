from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from dllm.http_auth import AuthPolicy, ClientKey
from dllm.request_scope import cache_scope
from dllm.scheduler import BatchedRuntime, Pending
from dllm.server import handler_for
from tests.test_scheduler import FakeRuntime, ChunkExecutor, PagedChunkExecutor


class CacheScopeTests(unittest.TestCase):
    def test_pending_captures_scope_before_worker_handoff(self):
        scope=object()
        token=cache_scope.set(scope)
        try:
            pending=Pending('a',[],1)
        finally:
            cache_scope.reset(token)
        self.assertIs(pending.cache_scope,scope)
        self.assertIsNone(Pending('a',[],1).cache_scope)

    def exercise_http(self,executor_type):
        alice='alice-cache-isolation-key'
        bob='bob-cache-isolation-key-1'
        with patch('dllm.scheduler.Runtime',FakeRuntime), \
                patch('dllm.prefix.available_memory',return_value=1024**3):
            runtime=BatchedRuntime({'a':'unused'},batch_size=2,queue_size=4,prefill_chunk_size=2,prefix_cache_tokens=32)
            executor=executor_type()
            executor.kv=[torch.zeros(2,2,1,256,2)]
            executor.lengths=[0,0]
            executor.release.set()
            runtime.runtime.executor=executor
            policy=AuthPolicy(clients=(ClientKey('alice',alice,2),ClientKey('bob',bob,2)))
            handler=handler_for(runtime,policy)
            observed=[]
            # The wrapper verifies reset in the same HTTP worker context,
            # including validation failures and completed streams.
            original=handler.do_POST
            def check_reset(instance):
                before=cache_scope.get()
                try:
                    return original(instance)
                finally:
                    observed.append((before,cache_scope.get()))
            handler.do_POST=check_reset
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            def request(key,prompt='abcd',stream=False,extra=None):
                body={'model':'a','messages':[{'role':'user','content':prompt}],'max_tokens':1,'stream':stream}
                if stream:body['stream_options']={'include_usage':True}
                if extra:body.update(extra)
                req=Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',json.dumps(body).encode(),
                            {'Content-Type':'application/json','Authorization':'Bearer '+key})
                try:response=urlopen(req,timeout=5)
                except HTTPError as error:response=error
                with response:return response.status,response.read().decode()
            def cached(key,prompt='abcd'):
                status,body=request(key,prompt)
                self.assertEqual(status,200,body)
                return json.loads(body)['dllm_timing']['cached_prompt_tokens']
            try:
                self.assertEqual(cached(alice),0)
                self.assertEqual(cached(alice),4)
                self.assertEqual(cached(bob),0)
                self.assertEqual(cached(bob),4)
                self.assertEqual(cached(alice,'abxy'),2)
                self.assertEqual(cached(bob,'abxy'),2) # Bob's own abcd snapshot.
                self.assertEqual(request(alice,extra={'cache_scope':'bob'})[0],400)
                self.assertEqual(request(alice,'fail')[0],400)
                with ThreadPoolExecutor(2) as pool:
                    results=list(pool.map(lambda key:request(key,'zzzz',True),(alice,bob)))
                for status,body in results:
                    self.assertEqual(status,200,body)
                    self.assertIn('data: [DONE]',body)
                    self.assertIn('"cached_prompt_tokens": 0',body)
                self.assertEqual(cached(alice,'zzzz'),4)
                self.assertEqual(cached(bob,'zzzz'),4)
                self.assertTrue(observed)
                self.assertTrue(all(before is None and after is None for before,after in observed))
            finally:
                server.shutdown();server.server_close();thread.join();runtime.close()
            if hasattr(executor,'pool'):
                self.assertEqual(executor.pool.used_tokens,0)

    def test_http_identity_survives_dense_scheduler_handoff(self):
        self.exercise_http(ChunkExecutor)

    def test_http_identity_survives_paged_scheduler_handoff(self):
        self.exercise_http(PagedChunkExecutor)
