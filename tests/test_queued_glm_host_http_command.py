from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import time
import threading
import unittest
from urllib.request import Request,urlopen
from dllm.cpu_snapshot_identity import fingerprint
from tests import test_glm_host_http_command as single


class QueuedGlmHostHttpCommandTests(unittest.TestCase):
    setUp=single.GlmHostHttpCommandTests.setUp
    snapshot=single.GlmHostHttpCommandTests.snapshot
    launch=single.GlmHostHttpCommandTests.launch
    event=single.GlmHostHttpCommandTests.event
    workers=single.GlmHostHttpCommandTests.workers
    finish_workers=single.GlmHostHttpCommandTests.finish_workers
    args=single.GlmHostHttpCommandTests.args
    post=single.GlmHostHttpCommandTests.post

    def test_actual_process_concurrent_json_sse_and_drain_acknowledges_stop(self):
        key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
        self.environment.update(DLLM_API_KEY=key,DLLM_ADMIN_API_KEY=admin)
        identity=fingerprint(self.path).model_id;common=['--verify-model-id','--model-id',identity]
        workers=self.workers('fp8_reference',common)
        coordinator=self.launch(self.args('serve',extra=common+['--queue-size','3','--queue-timeout','3','--stop-after-drain']))
        ready=self.event(coordinator);self.assertTrue(ready['queue_enabled']);self.assertEqual(ready['queue_capacity'],3)
        self.base=ready['url'];self.headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
        admin_headers=self.headers|{'Authorization':'Bearer '+admin}
        with self.post(stream=True,stream_options={'include_usage':True}) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire);self.assertIn('queue_seconds',wire)
        def request():
            with self.post(max_tokens=64) as response:return json.load(response)
        with ThreadPoolExecutor(3) as pool:
            futures=[pool.submit(request) for _ in range(3)]
            end=time.monotonic()+3
            while True:
                with urlopen(Request(self.base+'/v1/scheduler/status',headers=admin_headers),timeout=3) as response:status=json.load(response)
                count=status['worker_owned_generation_requests']+status['queued_generation_requests']
                if count==3 or time.monotonic()>=end:break
                threading.Event().wait(.005)
            self.assertEqual(count,3);self.assertEqual(status['batch_capacity'],1);self.assertTrue(status['client_admission_turns'])
            with urlopen(Request(self.base+'/v1/scheduler/drain',b'{}',admin_headers),timeout=3) as response:self.assertFalse(json.load(response)['drained'])
            results=[future.result(8) for future in futures]
        self.assertTrue(all(result['choices']==results[0]['choices'] and result['usage']==results[0]['usage'] for result in results))
        self.assertTrue(all(result['dllm_timing']['queue_seconds']>=0 for result in results))
        output,error=coordinator.communicate(timeout=8);self.assertEqual(coordinator.returncode,0,error)
        self.assertTrue(json.loads(output)['graceful_stop']);self.finish_workers(workers)
        self.assertNotIn(key,output+error);self.assertNotIn(admin,output+error)
