from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import torch
from dllm.host_stage_channel import HostStageChannel
from dllm.host_hybrid_worker import HostHybridWorker
from dllm.stage_protocol import StageState
from tests import test_host_tensor_channel,test_hybrid_stage


class HostHybridWorkerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_hybrid_stage.HybridStageTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        transports=test_host_tensor_channel.HostTensorChannelTests.pair(self,caps=(65536,65536),timeout=1)
        stages=[self.fixture.stage(0,2),self.fixture.stage(2,4)]
        self.workers=[HostHybridWorker(HostStageChannel(t,StageState(1,s.context,s.max_batch),vocab_size=s.c['vocab_size']),s)
                      for t,s in zip(transports,stages)]
        for worker in self.workers:self.addCleanup(worker.close)
        self.parallel([w.handshake for w in self.workers])

    def parallel(self,functions):
        with ThreadPoolExecutor(2) as pool:
            return [future.result(timeout=5) for future in [pool.submit(f) for f in functions]]

    def run_command(self,call):return self.parallel([call,self.workers[1].step])[0]

    def test_prefill_reordered_decode_layer_state_and_release_reuse(self):
        left,right=self.workers;full=self.fixture.fixture.fixture.executor()
        expected=full.prefill([1,2,3]).argmax(-1).tolist()
        self.assertEqual(self.run_command(lambda:left.prefill([1,2,3])),expected[0])
        expected=full.prefill([4,5],1).argmax(-1).tolist()
        self.assertEqual(self.run_command(lambda:left.prefill([4,5],1)),expected[0])
        for tokens,slots in (([6,7],[1,0]),([8],[0]),([9],[1])):
            _,ids=full.decode_batch(tokens,slots)
            self.assertEqual(self.run_command(lambda:left.decode(tokens,slots)),ids.tolist())
        for worker in self.workers:
            self.assertEqual(worker.stage.lengths,full.lengths)
            for name in ('kv','state','conv'):
                for layer,tensor in getattr(worker.stage,name).items():
                    torch.testing.assert_close(tensor,getattr(full,name)[layer],rtol=3e-5,atol=3e-6)
        self.run_command(lambda:left.release([0]))
        for worker in self.workers:
            for tensor in worker.stage.kv.values():self.assertEqual(int(tensor[:,0].count_nonzero()),0)
            for group in (worker.stage.conv,worker.stage.state):
                for tensor in group.values():self.assertEqual(int(tensor[0].count_nonzero()),0)
        expected=full.prefill([10,11],0).argmax(-1).tolist()
        self.assertEqual(self.run_command(lambda:left.prefill([10,11],0)),expected[0])
        self.assertEqual(left.channel.state.generations[0],2)
        self.run_command(left.ping);self.run_command(left.stop)
        self.assertTrue(all(w.closed and w.stage.closed for w in self.workers))
        self.assertTrue(all(not w.stage.w and not w.stage.kv and not w.stage.state for w in self.workers))

    def test_packed_prefill_matches_independent_slot_controls(self):
        left,_=self.workers;full=self.fixture.fixture.fixture.executor()
        expected=full.prefill_batch([[1,2],[3,4,5]],[1,0]).argmax(-1).tolist()
        self.assertEqual(self.run_command(lambda:left.prefill_batch([[1,2],[3,4,5]],[1,0])),expected)
        self.assertEqual(left.channel.state.lengths,[3,2])

    def test_invalid_inputs_do_not_send_or_poison_epoch(self):
        left,_=self.workers
        with patch.object(left.channel,'begin',side_effect=AssertionError('command sent')):
            for call in (lambda:left.prefill([-1]),lambda:left.prefill([1],True),lambda:left.decode([1],[0]),
                         lambda:left.prefill_batch([[1],[2]],[0,0]),lambda:left.release([0])):
                with self.assertRaises(ValueError):call()
        self.assertFalse(left.failed);self.assertFalse(left.channel.state.failed)

    def test_output_stage_failure_releases_both_owned_stages(self):
        left,right=self.workers
        with patch.object(right.stage,'forward_layers',side_effect=RuntimeError('output stage failed')),ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(left.prefill,[1,2]),pool.submit(right.step)]
            for future in futures:
                with self.assertRaises((RuntimeError,OSError,EOFError)):future.result(timeout=5)
        self.assertTrue(all(w.failed and w.closed and w.stage.closed for w in self.workers))
        self.assertTrue(all(w.channel.state.sequence==1 for w in self.workers))

    def test_close_failure_retains_stage_for_retry(self):
        left,_=self.workers
        with patch.object(left.stage,'synchronize',side_effect=RuntimeError('close failed')):
            with self.assertRaisesRegex(RuntimeError,'close failed'):left.close()
            self.assertFalse(left.closed);self.assertTrue(left.stage.w)
        left.close();self.assertTrue(left.closed);self.assertFalse(left.stage.w)

    def test_setup_rejects_layer_overlap_and_configuration_disagreement(self):
        for mismatch in ('range','config'):
            transports=test_host_tensor_channel.HostTensorChannelTests.pair(self,caps=(65536,65536),timeout=1)
            stages=[self.fixture.stage(0,2),self.fixture.stage(1 if mismatch=='range' else 2,4)]
            if mismatch=='config':stages[1].c=dict(stages[1].c,host_contract_test='different')
            workers=[HostHybridWorker(HostStageChannel(t,StageState(1,s.context,s.max_batch),vocab_size=s.c['vocab_size']),s)
                     for t,s in zip(transports,stages)]
            for worker in workers:self.addCleanup(worker.close)
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(w.handshake) for w in workers]
                for future in futures:
                    with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):future.result(timeout=5)
            self.assertTrue(all(w.failed and w.stage.closed for w in workers))

    def test_close_interrupts_idle_receiver_and_releases_stage(self):
        import threading
        import time
        right=self.workers[1];entered=threading.Event();begin=right.channel.begin
        def waiting(*args):entered.set();return begin(*args)
        with patch.object(right.channel,'begin',side_effect=waiting),ThreadPoolExecutor(1) as pool:
            future=pool.submit(right.step);self.assertTrue(entered.wait(1));started=time.monotonic();right.close()
            with self.assertRaises((RuntimeError,OSError,EOFError)):future.result(timeout=1)
            self.assertLess(time.monotonic()-started,1)
        self.assertTrue(right.closed and right.stage.closed)

    def test_primary_execution_error_survives_failed_stage_cleanup(self):
        left,right=self.workers
        with patch.object(right.stage,'forward_layers',side_effect=RuntimeError('primary compute failed')), \
                patch.object(right.stage,'close',side_effect=RuntimeError('secondary cleanup failed')),ThreadPoolExecutor(2) as pool:
            first=pool.submit(left.prefill,[1]);second=pool.submit(right.step)
            with self.assertRaisesRegex(RuntimeError,'primary compute failed') as error:second.result(timeout=5)
            with self.assertRaises((RuntimeError,OSError,EOFError)):first.result(timeout=5)
            self.assertTrue(error.exception.__notes__);self.assertFalse(right.closed);self.assertTrue(right.stage.w)
        right.close();self.assertTrue(right.closed);self.assertFalse(right.stage.w)
