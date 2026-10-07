import queue
import time
import unittest
from unittest.mock import patch

from dllm.distributed_serving import Request,DistributedRuntime
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from tests import test_host_serving as host
from tests.test_cpu_pipeline_generation import OPTIONS
from tests.test_distributed_serving import Pair


class HostInterleavedPrefillTests(unittest.TestCase):
    def setUp(self):
        self.fixture=host.HostServingTests();self.fixture.runtime_options=dict(interleaved_prefill=True)
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.runtime=self.fixture.runtime;self.pair=self.fixture.pair

    def request(self,tokens,maximum):
        return Request(tokens,maximum,time.monotonic(),queue.Queue(9))

    def result(self,request):
        events=[]
        while not request.events.empty():events.append(request.events.get_nowait())
        for event in events:
            if isinstance(event,Exception):raise event
        return events

    def check_control(self,request):
        source=load_cpu_checkpoint(self.fixture.fixture.path,max_context=16,quantization='q4_0')
        control=CpuPipelineReferenceGenerator(source,[2**22]*2,vocab_size=self.fixture.text.vocab_size,**OPTIONS)
        self.addCleanup(control.cleanup)
        expected=list(control.generate(request.tokens,max_new_tokens=request.maximum,chunk_size=1))
        result=self.result(request)
        self.assertEqual(result[-1]['usage'],expected[-1]['usage'])
        self.assertEqual(''.join(e.get('text','') for e in result),
            self.fixture.text.decode([e['token_id'] for e in expected if 'token_id' in e]))

    def test_unequal_prompts_round_robin_and_decode_before_long_prompt_finishes(self):
        short=self.request([1],5);long=self.request([2]*8,3)
        self.runtime.requests.put(short);self.runtime.requests.put(long)
        trace=[];chunk=self.pair.prefill_chunk;decode=self.pair.decode
        def prefill(tokens,slot,start):
            trace.append(('prefill',slot,start));return chunk(tokens,slot,start)
        def advance(tokens,slots):
            trace.append(('decode',tuple(slots),None));return decode(tokens,slots)
        with patch.object(self.pair,'prefill_chunk',side_effect=prefill),patch.object(self.pair,'decode',side_effect=advance):
            self.runtime.run(idle_timeout=.15,heartbeat_interval=.02)
        self.check_control(short);self.check_control(long)
        self.assertEqual([e[2] for e in trace if e[:2]==('prefill',1)],list(range(8)))
        first_long=trace.index(('prefill',1,0));last_long=trace.index(('prefill',1,7))
        self.assertTrue(any(e[0]=='decode' and 0 in e[1] for e in trace[first_long:last_long]))
        self.assertFalse(any(self.pair.state.lengths));self.assertEqual(self.runtime.completed,2)
        self.assertTrue(self.runtime.manage('status')['interleaved_prefill'])
        self.assertEqual(self.runtime.prefill_turn,0)

    def test_partial_cancel_releases_before_reuse_and_never_publishes_chunk_samples(self):
        cancelled=self.request([1]*8,3);survivor=self.request([2]*3,3);replacement=self.request([3]*2,2)
        for r in (cancelled,survivor):self.runtime.requests.put(r)
        original=self.pair.prefill_chunk;trace=[]
        def prefill(tokens,slot,start):
            trace.append((slot,start));token=original(tokens,slot,start)
            if slot==0 and start==0 and len(trace)==1:
                cancelled.cancelled.set();self.runtime.requests.put(replacement)
            return token
        with patch.object(self.pair,'prefill_chunk',side_effect=prefill):self.runtime.run(idle_timeout=.15)
        self.assertEqual(self.result(cancelled),[])
        self.check_control(survivor);self.check_control(replacement)
        self.assertEqual([start for slot,start in trace if slot==0],[0,0,1])
        self.assertEqual(self.runtime.completed,2);self.assertFalse(any(self.pair.state.lengths))

    def test_live_arrival_is_admitted_between_existing_decode_steps(self):
        existing=self.request([1],6);incoming=self.request([2]*7,2);self.runtime.requests.put(existing)
        original_decode=self.pair.decode;original_chunk=self.pair.prefill_chunk;trace=[];submitted=False
        def decode(tokens,slots):
            nonlocal submitted
            trace.append(('decode',tuple(slots)))
            result=original_decode(tokens,slots)
            if not submitted:self.runtime.requests.put(incoming);submitted=True
            return result
        def chunk(tokens,slot,start):
            trace.append(('chunk',slot,start));return original_chunk(tokens,slot,start)
        with patch.object(self.pair,'decode',side_effect=decode),patch.object(self.pair,'prefill_chunk',side_effect=chunk):
            self.runtime.run(idle_timeout=.15)
        self.check_control(existing);self.check_control(incoming)
        first=trace.index(('chunk',1,0));last=trace.index(('chunk',1,6))
        self.assertTrue(any(e[0]=='decode' and 0 in e[1] for e in trace[first:last]))
        self.assertFalse(any(self.pair.state.lengths))

    def test_partial_peer_loss_discards_session_without_success(self):
        request=self.request([1]*8,3);self.runtime.requests.put(request)
        original=self.pair.prefill_chunk
        def prefill(tokens,slot,start):
            token=original(tokens,slot,start)
            if start==0:self.fixture.workers[1].close()
            return token
        with patch.object(self.pair,'prefill_chunk',side_effect=prefill):
            with self.assertRaises((RuntimeError,OSError,EOFError)):self.runtime.run()
        with self.assertRaisesRegex(RuntimeError,'session failed'):self.result(request)
        self.assertTrue(self.fixture.workers[0].closed);self.assertEqual(self.runtime.completed,0)
        with self.assertRaises(ValueError):DistributedRuntime(Pair(),interleaved_prefill=True)
        with self.assertRaises(ValueError):DistributedRuntime(Pair(),interleaved_prefill=1)
