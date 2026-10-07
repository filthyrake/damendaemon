from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.cpu_host_worker_build import build_cpu_host_worker,CpuHostWorkerBuildError
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_hybrid_stage import CpuHybridStage
from dllm.cpu_pipeline_build import build_cpu_pipeline
from dllm.host_endpoint import HostPeerListener,connect_peer
from dllm.host_tensor_channel import HostTensorChannel
from tests.test_host_tls import contexts,FIXTURES
from dllm.host_tls import certificate_pin
from tests.test_cpu_hybrid_checkpoint import snapshot
from tests.test_cpu_stage_plan import OPTIONS


class CpuHostWorkerBuildTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name);snapshot(self.path)

    def transports(self):
        client,server=contexts();raw=socket.socket();self.addCleanup(raw.close);raw.bind(('127.0.0.1',0));address=raw.getsockname()
        listener=HostPeerListener(raw,server,peer_pin=certificate_pin(FIXTURES/'client.pem'));self.addCleanup(listener.close)
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(listener.accept,timeout=2)
            left=connect_peer(*address,client,peer_pin=certificate_pin(FIXTURES/'server.pem'),server_hostname='localhost',timeout=2)
            right=future.result(timeout=3)
        listener.close();channels=[HostTensorChannel(s,b'w'*32,rank,max_frame_bytes=65536,timeout=1) for rank,s in enumerate((left,right))]
        for c in channels:self.addCleanup(c.close)
        return channels

    def build(self,transport,mode='none'):
        result=build_cpu_host_worker(self.path,0 if transport.rank==0 else 2,2 if transport.rank==0 else 4,transport,
                                    epoch=1,max_context=16,quantization=mode,**OPTIONS)
        self.addCleanup(result.close);return result

    def together(self,functions):
        with ThreadPoolExecutor(2) as pool:return [f.result(timeout=5) for f in [pool.submit(call) for call in functions]]

    def test_authenticated_stage_local_workers_execute_against_whole_pipeline(self):
        for mode in ('q4_0','q8_0'):
            left,right=[self.build(t,mode) for t in self.transports()]
            self.assertNotIn('head',left.stage.w);self.assertNotIn('embed',right.stage.w)
            self.together([left.handshake,right.handshake])
            source=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
            control=build_cpu_pipeline(source,[2**22]*2,**OPTIONS);self.addCleanup(control.close)
            for tokens,slot,start in (([1,2],0,0),([3,4],0,2),([5,6,7],1,0)):
                expected=control.prefill_chunk(tokens,slot,start,True).argmax(-1).tolist()[0]
                self.assertEqual(self.together([lambda:left.prefill_chunk(tokens,slot,start),right.step])[0],expected)
            logits,ids=control.decode_batch([8,9],[1,0])
            self.assertEqual(self.together([lambda:left.decode([8,9],[1,0]),right.step])[0],ids.tolist())
            self.together([lambda:left.release([0,1]),right.step])
            expected=control.prefill_batch([[1,2],[3]],[0,1]).argmax(-1).tolist()
            self.assertEqual(self.together([lambda:left.prefill_batch([[1,2],[3]],[0,1]),right.step])[0],expected)
            self.together([left.stop,right.step]);self.assertTrue(left.closed and right.closed)
            for worker in (left,right):self.assertFalse(worker.stage.w);self.assertFalse(worker.stage.model.scales)

    def test_bad_control_leaves_transport_caller_owned(self):
        left,_=self.transports()
        for options in (dict(epoch=0),dict(max_context=513),dict(max_batch=5)):
            with patch('dllm.cpu_host_worker_build.load_cpu_stage_checkpoint',side_effect=AssertionError('loaded')):
                with self.assertRaises(ValueError):build_cpu_host_worker(self.path,0,2,left,**(dict(epoch=1)|options))
            self.assertFalse(left.closed);self.assertIsNone(getattr(left,'_cpu_worker_builder_owner',None))
        with self.assertRaises(ValueError):build_cpu_host_worker(self.path,1,2,left,epoch=1,max_context=16)
        self.assertFalse(left.closed)

    def test_load_failure_and_stage_partial_failure_close_transport(self):
        for kind in ('load','stage'):
            left,_=self.transports();initialize=CpuHybridStage.__init__;stages=[]
            def fail(stage,*args,**kwargs):
                stages.append(stage);initialize(stage,*args,**kwargs);raise MemoryError('partial stage')
            target='dllm.cpu_host_worker_build.load_cpu_stage_checkpoint' if kind=='load' else 'dllm.cpu_hybrid_stage.CpuHybridStage.__init__'
            with patch(target,side_effect=ValueError('invalid source')) if kind=='load' else patch(target,fail):
                with self.assertRaises(CpuHostWorkerBuildError) as caught:self.build(left)
            self.assertTrue(caught.exception.closed);self.assertTrue(left.closed)
            for stage in stages:self.assertFalse(stage.w);self.assertFalse(stage.model.scales)

    def test_failed_partial_cleanup_retains_owner_and_primary(self):
        left,_=self.transports();initialize=CpuHybridStage.__init__
        def fail(stage,*args,**kwargs):initialize(stage,*args,**kwargs);raise RuntimeError('primary stage failure')
        with patch.object(CpuHybridStage,'__init__',fail),patch.object(CpuHybridStage,'release_partial',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(CpuHostWorkerBuildError,'primary stage failure') as caught:self.build(left)
        failure=caught.exception;self.assertFalse(failure.closed);self.assertTrue(failure.stage.w);self.assertTrue(left.closed)
        self.assertTrue(failure.__notes__);failure.close();self.assertTrue(failure.closed);self.assertFalse(failure.stage.w)

    def test_setup_mismatch_and_duplicate_transport_claim(self):
        transports=self.transports();left=self.build(transports[0],'q4_0');right=self.build(transports[1],'q8_0')
        with self.assertRaises(ValueError):self.build(transports[0])
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(w.handshake) for w in (left,right)]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):future.result(timeout=5)
        self.assertTrue(left.closed and right.closed)
