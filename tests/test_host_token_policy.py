from concurrent.futures import ThreadPoolExecutor
import json
import unittest
from unittest.mock import patch
import torch
from dllm.cpu_host_worker_build import build_cpu_host_worker
from dllm.host_hybrid_worker import HostHybridWorker
from dllm.host_stage_channel import HostStageChannel
from dllm.stage_protocol import StageState
from dllm.cpu_hybrid_checkpoint import load_cpu_stage_checkpoint
from dllm.cpu_hybrid_stage import CpuHybridStage
from tests import test_cpu_host_worker_build as fixture_module
from tests.test_cpu_stage_plan import OPTIONS


class HostTokenPolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.CpuHostWorkerBuildTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        p=self.fixture.path/'config.json';c=json.loads(p.read_text());c['image_token_id']=8;p.write_text(json.dumps(c))

    def workers(self,policies=None):
        policies=policies or [dict(mapped_vocab_size=60,excluded_ids=[9])]*2
        workers=[]
        for rank,transport in enumerate(self.fixture.transports()):
            worker=build_cpu_host_worker(self.fixture.path,0 if rank==0 else 2,2 if rank==0 else 4,transport,
                        epoch=1,max_context=16,quantization='q4_0',**OPTIONS,**policies[rank])
            self.addCleanup(worker.close);workers.append(worker)
        return workers

    def together(self,functions):return self.fixture.together(functions)

    def test_high_media_and_padded_logits_never_sample_on_any_path(self):
        left,right=self.workers();self.together([left.handshake,right.handshake])
        def logits(hidden):
            value=torch.zeros(len(hidden),64);value[:,63]=300;value[:,8]=200;value[:,9]=100;value[:,7]=90;return value
        with patch.object(right.stage,'output_logits',logits):
            self.assertEqual(self.together([lambda:left.prefill_chunk([1,2]),right.step])[0],7)
            self.assertEqual(self.together([lambda:left.prefill_chunk([3],0,2),right.step])[0],7)
            self.assertEqual(self.together([lambda:left.prefill([4],1),right.step])[0],7)
            self.assertEqual(self.together([lambda:left.decode([5,6],[1,0]),right.step])[0],[7,7])
            self.together([lambda:left.release([0,1]),right.step])
            self.assertEqual(self.together([lambda:left.prefill_batch([[1],[2,3]],[0,1]),right.step])[0],[7,7])
        with patch.object(left.channel,'begin',side_effect=AssertionError('sent')):
            for tokens in ([8],[9],[60],[63]):
                with self.assertRaises(ValueError):left.prefill(tokens)
        self.assertFalse(left.failed)

    def test_default_policy_also_suppresses_configured_media(self):
        left,right=self.workers([{},{}]);self.together([left.handshake,right.handshake])
        value=torch.zeros(1,64);value[0,8]=100;value[0,7]=90
        with patch.object(right.stage,'output_logits',return_value=value):
            self.assertEqual(self.together([lambda:left.prefill([1]),right.step])[0],7)

    def test_peer_policy_mismatch_rejects_setup_and_releases(self):
        for other in (dict(mapped_vocab_size=59,excluded_ids=[9]),dict(mapped_vocab_size=60,excluded_ids=[10])):
            workers=self.workers([dict(mapped_vocab_size=60,excluded_ids=[9]),other])
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(w.handshake) for w in workers]
                for f in futures:
                    with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):f.result(timeout=5)
            self.assertTrue(all(w.closed and not w.stage.w for w in workers))

    def test_bad_policy_does_not_claim_prepared_resources(self):
        transport,_=self.fixture.transports()
        model=load_cpu_stage_checkpoint(self.fixture.path,0,2,max_context=16)
        stage=CpuHybridStage(model,0,2,**OPTIONS);self.addCleanup(stage.close)
        channel=HostStageChannel(transport,StageState(1,16,2),vocab_size=64);self.addCleanup(channel.close)
        for policy in (dict(mapped_vocab_size=True),dict(mapped_vocab_size=65),dict(mapped_vocab_size=0),
                       dict(excluded_ids=[True]),dict(excluded_ids=[64]),dict(mapped_vocab_size=2,excluded_ids=[0,1])):
            with self.assertRaises(ValueError):HostHybridWorker(channel,stage,**policy)
            self.assertIsNone(getattr(stage,'_pipeline_owner',None));self.assertIsNone(getattr(channel,'_worker_owner',None))
            self.assertFalse(channel.closed);self.assertFalse(stage.closed)

    def test_untrusted_result_policy_violation_poison_and_cleanup(self):
        left,right=self.workers();self.together([left.handshake,right.handshake])
        original=left.channel.complete_tokens
        def corrupt(command,*args):original(command,*args);return torch.tensor([63],dtype=torch.int64)
        with patch.object(left.channel,'complete_tokens',corrupt),ThreadPoolExecutor(2) as pool:
            a=pool.submit(left.prefill,[1]);b=pool.submit(right.step)
            with self.assertRaisesRegex(ValueError,'unmapped'):a.result(timeout=5)
            b.result(timeout=5)
        self.assertTrue(left.failed and left.closed);self.assertTrue(left.channel.state.failed);self.assertFalse(left.stage.w)
