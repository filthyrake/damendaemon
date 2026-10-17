import contextlib
import io
import queue
import time
import unittest
from unittest.mock import Mock,patch

from damendaemon.distributed_serving import DistributedRuntime,Request
from damendaemon.request_scope import CacheScope
from tests.test_distributed_serving import BatchedPair


class DistributedDecodeTurnsTests(unittest.TestCase):
    def execute(self,enabled,scopes):
        pair=BatchedPair();runtime=DistributedRuntime(pair,client_decode_turns=enabled)
        requests=[Request([index+1],6,time.monotonic(),queue.Queue(9),cache_scope=scope) for index,scope in enumerate(scopes)]
        for request in requests:runtime.requests.put(request)
        runtime.run(idle_timeout=.15)
        outputs=[]
        for request in requests:
            events=[]
            while not request.events.empty():events.append(request.events.get_nowait())
            outputs.append(''.join(e.get('text','') for e in events))
            self.assertEqual(events[-1]['usage']['completion_tokens'],6)
        self.assertEqual(outputs,['ABCDEF']*len(scopes));self.assertFalse(any(pair.state.lengths))
        self.assertFalse(runtime.decode_turns.recent)
        self.assertEqual(runtime.manage('status')['client_decode_turns'],enabled)
        return [c[2] for c in pair.calls if c[0]=='decode']

    def test_opt_in_rotates_unequal_clients_without_changing_outputs(self):
        a,b=CacheScope(),CacheScope()
        calls=self.execute(True,[a,a,b])
        self.assertEqual(calls[:2],[[0,1],[1,2]])
        self.assertTrue(all(len(c)==2 for c in calls[:5]))
        self.assertEqual(self.execute(False,[a,a,b]),[[0,1,2]]*5)
        self.assertEqual(self.execute(True,[None,None,None]),[[0,1,2]]*5)
        with self.assertRaises(ValueError):DistributedRuntime(BatchedPair(),client_decode_turns=1)

    def test_gpu_command_forwards_opt_in_without_constructing_hardware(self):
        from damendaemon.distributed_server_cli import main
        for enabled in (False,True):
            pair=Mock(rank=0)
            with patch.dict('os.environ',{},clear=False),patch('damendaemon.distributed.StagePair',return_value=pair), \
                    patch('damendaemon.distributed_serving.DistributedRuntime') as runtime, \
                    patch('damendaemon.server.BoundedHTTPServer'),patch('damendaemon.distributed_server_cli.threading.Thread'), \
                    patch('damendaemon.distributed_server_cli.signal.signal'),contextlib.redirect_stdout(io.StringIO()):
                main(['/local',*(['--client-decode-turns'] if enabled else [])])
                self.assertEqual(runtime.call_args.kwargs,dict(client_decode_turns=enabled))
