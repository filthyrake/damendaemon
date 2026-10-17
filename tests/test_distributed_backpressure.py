from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch

from damendaemon.distributed_serving import DistributedRuntime,Request,STREAM_BUFFER_EVENTS
from tests.test_distributed_serving import BatchedPair,Pair,RuntimeHelpers


class BackpressureTests(RuntimeHelpers,unittest.TestCase):
    def wait_for(self,condition):
        deadline=time.monotonic()+2
        while not condition() and time.monotonic()<deadline:time.sleep(.002)
        self.assertTrue(condition())

    def stream(self,runtime,maximum=20):
        requests=[]
        def capture(*args,**kwargs):
            request=Request(*args,**kwargs);requests.append(request);return request
        stream=runtime.generate('model',None,maximum,raw_prompt=[1])
        self.addCleanup(stream.close)
        with patch('damendaemon.distributed_serving.Request',side_effect=capture):first=next(stream)
        return stream,requests[0],first

    def test_slow_client_pauses_its_slot_while_other_clients_complete(self):
        pair=BatchedPair();runtime=DistributedRuntime(pair,client_decode_turns=True);_,errors=self.start(runtime,heartbeat_interval=.02)
        stream,request,first=self.stream(runtime)
        self.wait_for(lambda:request.events.qsize()==STREAM_BUFFER_EVENTS)
        before=pair.state.lengths[0];self.assertLess(before,20)
        result=list(runtime.generate('model',None,4,raw_prompt=[2]))
        self.assertEqual(''.join(e.get('text','') for e in result),'ABCD')
        self.assertEqual(pair.state.lengths[0],before)
        self.wait_for(lambda:runtime.heartbeats>0)
        self.assertEqual(pair.state.lengths[0],before)
        remaining=list(stream)
        self.assertEqual(first['text']+''.join(e.get('text','') for e in remaining),'ABCDEFGHIJKLMNOPQRST')
        self.assertEqual(remaining[-1]['usage']['completion_tokens'],20)
        self.assertEqual(request.events.maxsize,9);self.assertEqual(pair.state.lengths,[0]*4)
        self.assertEqual(errors,[])

    def test_reserved_terminal_capacity_releases_even_if_consumer_stops(self):
        pair=Pair();runtime=DistributedRuntime(pair);_,errors=self.start(runtime)
        stream,request,first=self.stream(runtime,9)
        self.wait_for(lambda:runtime.completed==1)
        self.assertEqual(request.events.qsize(),9);self.assertEqual(pair.state.lengths,[0])
        following=list(runtime.generate('model',None,2,raw_prompt=[2]))
        self.assertEqual(following[-1]['usage']['completion_tokens'],2)
        result=list(stream);self.assertEqual(result[-1]['usage']['completion_tokens'],9)
        self.assertEqual(first['text']+''.join(e.get('text','') for e in result),'ABCDEFGHI')
        self.assertEqual(errors,[])

    def test_full_buffer_cancel_releases_and_reuses_slot(self):
        pair=Pair();runtime=DistributedRuntime(pair);_,errors=self.start(runtime)
        stream,request,_=self.stream(runtime)
        self.wait_for(lambda:request.events.qsize()==STREAM_BUFFER_EVENTS)
        stream.close();self.wait_for(lambda:pair.releases==1)
        result=list(runtime.generate('model',None,2,raw_prompt=[2]))
        self.assertEqual(result[-1]['usage']['completion_tokens'],2)
        self.assertEqual(pair.releases,2);self.assertEqual(runtime.completed,1);self.assertEqual(errors,[])

    def test_full_buffer_peer_failure_delivers_error_without_blocking_owner(self):
        pair=BatchedPair();runtime=DistributedRuntime(pair);owner,errors=self.start(runtime)
        stream,request,_=self.stream(runtime)
        self.wait_for(lambda:request.events.qsize()==STREAM_BUFFER_EVENTS)
        pair.fail=True
        with self.assertRaisesRegex(RuntimeError,'session failed'):
            list(runtime.generate('model',None,3,raw_prompt=[2]))
        owner.join(2);self.assertFalse(owner.is_alive());self.assertEqual(len(errors),1)
        self.assertLessEqual(request.events.qsize(),9)
        with self.assertRaisesRegex(RuntimeError,'session failed'):list(stream)
        self.assertTrue(runtime.closed.is_set())
