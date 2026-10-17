from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import unittest
from urllib.request import Request,urlopen
from damendaemon.cpu_snapshot_identity import fingerprint
from tests import test_glm_flash_image_host_command as image,test_glm_flash_host_http_command as text
from tests.test_glm_flash_inline_images import messages
from tests.test_glm_flash_tool_generation import TOOLS


class FlashImageHostHttpCommandTests(unittest.TestCase):
    setUp=image.FlashImageHostCommandTests.setUp
    snapshot=image.FlashImageHostCommandTests.snapshot
    args=image.FlashImageHostCommandTests.args
    launch=image.FlashImageHostCommandTests.launch
    event=image.FlashImageHostCommandTests.event
    workers=image.FlashImageHostCommandTests.workers
    finish_workers=image.FlashImageHostCommandTests.finish_workers
    service=text.FlashHostHttpCommandTests.service
    test_verified_http_idle_signal=text.FlashHostHttpCommandTests.test_actual_verified_http_json_sse_auth_heartbeat_and_idle_sigterm_stop
    test_invalid_serving_preflight=text.FlashHostHttpCommandTests.test_bad_serving_options_fail_before_network
    test_peer_loss_terminal=text.FlashHostHttpCommandTests.test_peer_loss_stops_http_without_graceful_result
    test_active_signal_terminal=text.FlashHostHttpCommandTests.test_active_sigterm_cancels_sse_and_discards_epoch

    def post(self,**options):
        return text.FlashHostHttpCommandTests.post(self,**(dict(messages=messages())|options))

    def test_queued_image_tool_concurrency_drains_and_stops_all_processes(self):
        key=secrets.token_urlsafe(32);admin=secrets.token_urlsafe(32)
        self.environment.update(DAMENDAEMON_API_KEY=key,DAMENDAEMON_ADMIN_API_KEY=admin)
        identity=fingerprint(self.path).model_id;common=['--verify-model-id','--model-id',identity]
        workers=self.workers('fp8_reference',common)
        coordinator=self.launch(self.args('serve',extra=common+['--queue-size','4','--stop-after-drain']))
        ready=self.event(coordinator);self.assertEqual(ready['event'],'http_ready')
        self.assertEqual(ready['modality'],'still_image');self.assertTrue(ready['vision_identity_verified']);self.assertTrue(ready['queue_enabled'])
        self.base=ready['url'];self.headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
        self.admin=self.headers|{'Authorization':'Bearer '+admin}
        def request():
            with self.post(tools=TOOLS,max_tokens=3) as response:return json.load(response)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda _:request(),range(3)))
        self.assertTrue(all(result['usage']==results[0]['usage'] for result in results))
        self.assertTrue(all(result['damendaemon_timing']['queue_seconds']>=0 for result in results))
        with urlopen(Request(self.base+'/v1/models/status',headers=self.admin),timeout=3) as response:status=json.load(response)
        self.assertEqual(status['modality'],'still_image');self.assertTrue(status['vision_identity_verified'])
        with urlopen(Request(self.base+'/v1/scheduler/drain',b'{}',self.admin),timeout=3) as response:self.assertTrue(json.load(response)['drained'])
        output,error=coordinator.communicate(timeout=8);self.assertEqual(coordinator.returncode,0,error)
        self.assertTrue(json.loads(output)['graceful_stop']);self.finish_workers(workers)
        self.assertNotIn(key,output+error);self.assertNotIn(admin,output+error)
