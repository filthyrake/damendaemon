from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import torch
from dllm.host_topology import HostTopology
from dllm.planned_host_stage_channel import PlannedHostStageChannel
from dllm.host_tensor_channel import HostTensorChannel
from dllm.stage_protocol import StageState,Command,PREFILL,PREFILL_CHUNK,PREFILL_BATCH,DECODE,RELEASE,PING,STOP
from tests.test_cpu_hybrid_stage import model
from tests import test_host_tls


class PlannedHostStageTests(unittest.TestCase):
    def pair(self,*,index=1,identities=(b'm'*32,b'm'*32),ready=True):
        fixture=test_host_tls.HostTlsTests();self.addCleanup(fixture.doCleanups)
        sockets=fixture.pair();config=model().config;channels=[]
        for rank,sock in enumerate(sockets):
            plan=HostTopology(config,[(0,1),(1,2),(2,4)],identities[rank],context=16,chunk_size=8,
                frame_bytes=4096,mapped_vocab_size=60,excluded_ids=[59])
            transport=HostTensorChannel(sock,plan.stage_identity(index),rank,max_frame_bytes=4096,timeout=.3)
            self.addCleanup(transport.close)
            channel=PlannedHostStageChannel(transport,StageState(7,16),plan,index)
            self.addCleanup(channel.close);channels.append(channel)
        if ready:self.together([c.handshake for c in channels])
        return channels

    def together(self,functions,*,fail=False):
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(fn) for fn in functions];results=[]
            for future in futures:
                if fail:
                    with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):future.result(3)
                else:results.append(future.result(3))
            return results

    def begin(self,channels,command,*,fail=False):
        return self.together([lambda:channels[0].begin(command),channels[1].begin],fail=fail)

    def test_tls_intermediate_activations_chunks_decode_release_and_reuse(self):
        channels=self.pair();commands=[Command(7,1,PREFILL_CHUNK,3,((0,1,0),)),
            Command(7,2,PREFILL_CHUNK,2,((0,1,3),)),Command(7,3,DECODE,1,((0,1,5),)),
            Command(7,4,RELEASE,0,((0,1,6),)),Command(7,5,PREFILL_BATCH,5,((0,2,2),(1,1,3))),
            Command(7,6,PING,0,()),Command(7,7,STOP,0,())]
        for command in commands:
            self.assertEqual(self.begin(channels,command),[command,command])
            if command.rows:
                values=torch.arange(command.rows*32,dtype=torch.float32).reshape(command.rows,32)
                received=self.together([lambda:channels[0].complete_activation(command),lambda:channels[1].complete_activation(command,values)])
                torch.testing.assert_close(received[0],values,rtol=0,atol=0)
                saved=received[0].clone();values.zero_();torch.testing.assert_close(received[0],saved,rtol=0,atol=0)
            else:self.together([lambda c=c:c.finish(command) for c in channels])
            self.assertEqual(channels[0].state.lengths,channels[1].state.lengths)
            self.assertTrue(all(c.state.sequence==command.sequence+1 for c in channels))
        self.assertTrue(all(c.state.closed and not c.state.failed for c in channels))

    def test_acknowledgment_precedes_activation_cursor_commit(self):
        channels=self.pair();command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
        receive=channels[0].transport.receive;seen=[]
        def checked(shape,dtype):
            self.assertEqual(channels[0].state.lengths[0],0);self.assertEqual(channels[0].state.sequence,1)
            seen.append(shape);return receive(shape,dtype)
        with patch.object(channels[0].transport,'receive',side_effect=checked):
            self.together([lambda:channels[0].complete_activation(command),lambda:channels[1].complete_activation(command,torch.ones(3,32))])
        self.assertEqual(seen,[(3,32),(4,)]);self.assertEqual(channels[0].state.lengths[0],3)

    def test_final_stage_retains_tokens_and_rejects_excluded_output(self):
        channels=self.pair(index=2);command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
        result=self.together([lambda:channels[0].complete_tokens(command),lambda:channels[1].complete_tokens(command,torch.tensor([5]))])
        self.assertEqual(result[0].tolist(),[5])
        for token in (59,60):
            channels=self.pair(index=2);self.begin(channels,command)
            self.together([lambda:channels[0].complete_tokens(command),lambda:channels[1].complete_tokens(command,torch.tensor([token]))],fail=True)
            self.assertTrue(all(c.state.failed for c in channels));self.assertEqual(channels[0].state.lengths[0],0)

    def test_bad_contract_mode_shape_finiteness_or_budget_poison_epoch(self):
        channels=self.pair(identities=(b'a'*32,b'b'*32),ready=False)
        self.together([c.handshake for c in channels],fail=True)
        for bad in (torch.ones(2,32),torch.ones(3,32,dtype=torch.float64),torch.full((3,32),float('nan'))):
            channels=self.pair();command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
            self.together([lambda:channels[0].complete_activation(command),lambda:channels[1].complete_activation(command,bad)],fail=True)
            self.assertTrue(all(c.state.failed for c in channels));self.assertEqual(channels[0].state.lengths[0],0)
        channels=self.pair();self.begin(channels,Command(7,1,PREFILL,9,((0,1,0),)),fail=True)
        self.assertTrue(all(c.state.failed for c in channels))
        channels=self.pair();command=Command(7,1,PREFILL,1,((0,1,0),));self.begin(channels,command)
        self.together([lambda:channels[0].complete_activation(command),lambda:channels[1].complete_tokens(command,torch.tensor([1]))],fail=True)
        self.assertTrue(all(c.state.failed for c in channels))

    def test_corrupted_wire_result_or_ack_never_commits_coordinator(self):
        for corrupt in ('activation','ack'):
            channels=self.pair();command=Command(7,1,PREFILL,3,((0,1,0),));self.begin(channels,command)
            send=channels[1].transport.send
            def changed(tensor):
                if corrupt=='activation' and tensor.dtype==torch.float32:
                    tensor=tensor.clone();tensor[0,0]=float('inf')
                elif corrupt=='ack' and tensor.dtype==torch.int64 and tensor.numel()==4:
                    tensor=tensor.clone();tensor[2]+=1
                return send(tensor)
            with patch.object(channels[1].transport,'send',side_effect=changed),ThreadPoolExecutor(2) as pool:
                left=pool.submit(channels[0].complete_activation,command)
                right=pool.submit(channels[1].complete_activation,command,torch.ones(3,32))
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):left.result(3)
                try:right.result(3)
                except (ValueError,RuntimeError,OSError,EOFError):pass
            self.assertTrue(channels[0].state.failed);self.assertEqual(channels[0].state.sequence,1)
            self.assertEqual(channels[0].state.lengths[0],0)

    def test_close_interrupts_pending_result_without_commit(self):
        channels=self.pair();command=Command(7,1,PREFILL,1,((0,1,0),));self.begin(channels,command)
        with ThreadPoolExecutor(1) as pool:
            pending=pool.submit(channels[0].complete_activation,command);channels[0].close()
            with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):pending.result(2)
        self.assertTrue(channels[0].state.failed);self.assertEqual(channels[0].state.lengths[0],0)
