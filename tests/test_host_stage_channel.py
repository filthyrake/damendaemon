from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import patch
import torch
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.stage_protocol import StageState,Command,PREFILL,PREFILL_BATCH,DECODE,RELEASE,PING,STOP
from tests import test_host_tensor_channel


class HostStageChannelTests(unittest.TestCase):
    def pair(self,*,contexts=(32,32),vocabs=(100,100),timeout=.5):
        transport=test_host_tensor_channel.HostTensorChannelTests.pair(self,timeout=timeout)
        channels=[HostStageChannel(t,StageState(7,context),vocab_size=vocab)
                  for t,context,vocab in zip(transport,contexts,vocabs)]
        for channel in channels:self.addCleanup(channel.close)
        return channels

    def together(self,functions):
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(function) for function in functions]
            return [f.result(timeout=3) for f in futures]

    def ready(self,**options):
        channels=self.pair(**options);self.together([c.handshake for c in channels]);return channels

    def begin(self,channels,command):
        return self.together([lambda:channels[0].begin(command),channels[1].begin])

    def complete(self,channels,command,tokens):
        return self.together([lambda:channels[0].complete_tokens(command),lambda:channels[1].complete_tokens(command,tokens)])

    def failures(self,functions):
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(f) for f in functions]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):future.result(timeout=3)

    def test_prefill_decode_packed_release_slot_reuse_and_stop(self):
        channels=self.ready()
        commands=[Command(7,1,PREFILL,3,((0,1,0),)),Command(7,2,DECODE,1,((0,1,3),)),
                  Command(7,3,RELEASE,0,((0,1,4),)),Command(7,4,PREFILL_BATCH,5,((0,2,2),(1,1,3))),
                  Command(7,5,PING,0,()),Command(7,6,STOP,0,())]
        for command in commands:
            self.assertEqual(self.begin(channels,command),[command,command])
            if command.operation in (PREFILL,DECODE,PREFILL_BATCH):
                tokens=torch.tensor([12,13] if command.operation==PREFILL_BATCH else [12])
                results=self.complete(channels,command,tokens)
                for result in results:torch.testing.assert_close(result,tokens,rtol=0,atol=0)
            else:self.together([lambda c=c:c.finish(command) for c in channels])
            self.assertEqual(channels[0].state.lengths,channels[1].state.lengths)
            self.assertEqual(channels[0].state.generations,channels[1].state.generations)
            self.assertTrue(all(c.state.sequence==command.sequence+1 for c in channels))
        self.assertTrue(all(c.closed and c.state.closed and not c.state.failed for c in channels))
        self.assertEqual(channels[0].state.generations[:2],[2,1])

    def test_setup_bounds_disagreement_closes_both_epochs(self):
        for options in ({'contexts':(32,31)},{'vocabs':(100,101)}):
            channels=self.pair(**options);self.failures([c.handshake for c in channels])
            self.assertTrue(all(c.state.failed and c.closed for c in channels))

    def test_receiver_live_slot_rejection_precedes_activation(self):
        channels=self.ready();command=Command(7,1,PREFILL,3,((0,1,0),))
        # Simulate inconsistent receiver ownership; no activation may start.
        channels[1].state.lengths[0]=2
        self.failures([lambda:channels[0].begin(command),channels[1].begin])
        self.assertTrue(all(c.state.failed and c.state.pending is None for c in channels))
        self.assertEqual(channels[0].state.sequence,1)

    def test_coordinator_observes_completion_ack_before_commit(self):
        channels=self.ready();command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
        receive=channels[0].transport.receive;observed=[]
        def checked(shape,dtype):
            self.assertEqual(channels[0].state.lengths[0],0)
            self.assertEqual(channels[0].state.sequence,1)
            observed.append(tuple(shape));return receive(shape,dtype)
        with patch.object(channels[0].transport,'receive',side_effect=checked):self.complete(channels,command,torch.tensor([3]))
        self.assertEqual(observed,[(1,),(4,)]);self.assertEqual(channels[0].state.lengths[0],3)

    def test_wrong_ack_or_invalid_result_poisons_without_coordinator_commit(self):
        for failure in ('phase','sequence','token'):
            channels=self.ready();command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
            send=channels[1].transport.send
            def corrupt(tensor):
                value=tensor.clone()
                if failure=='token' and tensor.shape==(1,):value[0]=100
                if failure!='token' and tensor.shape==(4,):value[2 if failure=='phase' else 1]+=1
                send(value)
            with patch.object(channels[1].transport,'send',side_effect=corrupt),ThreadPoolExecutor(2) as pool:
                worker=pool.submit(channels[1].complete_tokens,command,torch.tensor([3]))
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):channels[0].complete_tokens(command)
                # A worker can have sent its final ack before the coordinator
                # detects corruption. There is no distributed rollback claim.
                try:worker.result(timeout=3)
                except (ValueError,RuntimeError,OSError,EOFError):pass
            self.assertTrue(channels[0].state.failed);self.assertEqual(channels[0].state.sequence,1)
            if not channels[1].state.failed:
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):channels[1].begin()
            self.assertTrue(channels[1].state.failed)

    def test_completion_timeout_and_close_cancel_blocked_receive(self):
        channels=self.ready(timeout=.05);command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
        with self.assertRaises(TimeoutError):channels[0].complete_tokens(command)
        self.assertTrue(channels[0].state.failed);self.assertEqual(channels[0].state.sequence,1)
        channels=self.ready(timeout=10);command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
        entered=threading.Event();receive=channels[0].transport.receive
        def waiting(*args):entered.set();return receive(*args)
        with patch.object(channels[0].transport,'receive',side_effect=waiting),ThreadPoolExecutor(1) as pool:
            future=pool.submit(channels[0].complete_tokens,command);self.assertTrue(entered.wait(1))
            started=time.monotonic();channels[0].close()
            with self.assertRaises((RuntimeError,OSError,EOFError)):future.result(timeout=1)
            self.assertLess(time.monotonic()-started,1)
        self.assertTrue(channels[0].state.failed);self.assertEqual(channels[0].state.sequence,1)
