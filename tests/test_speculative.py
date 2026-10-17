import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from damendaemon.sampling import Sampling,Sampler
from damendaemon.scheduler import BatchedRuntime,Pending
from damendaemon.speculative import configured_width,eligible,propose,create


class SpeculativeTests(unittest.TestCase):
    def test_configuration_and_bounded_proposals(self):
        for value in ('0','2','4','8'):
            with patch.dict(os.environ,{'DAMENDAEMON_SPECULATIVE_TOKENS':value}):self.assertEqual(configured_width(),int(value))
        for value in ('1','3','9','-1','garbage'):
            with patch.dict(os.environ,{'DAMENDAEMON_SPECULATIVE_TOKENS':value}),self.assertRaises(ValueError):configured_width()
        self.assertEqual(propose([1,2,3,4,1,2],2),[3,4])
        self.assertEqual(propose([1,2,3,4],3),[])
        self.assertEqual(propose([1,2,3]+[8]*1024+[1,2],3),[])

    def test_llama_verifier_requires_explicit_qualified_precision(self):
        executor=SimpleNamespace(device='cuda',kernels=object(),c={'model_type':'llama'},
            llama_precision='mixed',model=SimpleNamespace(scales={}),context=128)
        with patch.dict(os.environ,{'DAMENDAEMON_SPECULATIVE_TOKENS':'8'}),patch('damendaemon.speculative.Verifier') as verifier:
            for policy in ('standard','mixed'):
                executor.llama_precision=policy
                self.assertIsNone(create(executor,True))
            verifier.assert_not_called()
            executor.llama_precision='float32'
            self.assertIs(create(executor,True),verifier.return_value)
            verifier.assert_called_once_with(executor,8)
            verifier.reset_mock()
            executor.pool=object()
            self.assertIsNone(create(executor,True))
            del executor.pool
            executor.model.scales={'weight':1}
            self.assertIsNone(create(executor,True))
            executor.model.scales={}
            executor.model.packed_formats={'head':(256,((0,32,12,0),))}
            for family in ('llama','qwen2','qwen3'):
                executor.c['model_type']=family
                self.assertIsNone(create(executor,True))
            verifier.assert_not_called()

    def test_sampling_constraints_and_tools_are_ineligible(self):
        r=Pending('a',[],16);self.assertTrue(eligible(r))
        for kwargs in ({'temperature':.5},{'frequency_penalty':1},{'logit_bias':{1:1}}):
            r.sampler=Sampler(Sampling(**kwargs));self.assertFalse(eligible(r))
        r.sampler=Sampler(Sampling())
        for field,value in (('tools',[{}]),('tool_json',True),('schema','json'),('stops',('stop',))):
            previous=getattr(r,field);setattr(r,field,value);self.assertFalse(eligible(r));setattr(r,field,previous)
        r.sampler.logprobs_top=0;self.assertFalse(eligible(r))
        r.sampler.logprobs_top=None;r.sampler.constraint=object();self.assertFalse(eligible(r))
        r.sampler.constraint=None;r.messages=[{'role':'tool','content':'result'}];self.assertFalse(eligible(r))

    def setup_step(self):
        runtime=BatchedRuntime.__new__(BatchedRuntime);runtime.condition=threading.Condition();runtime.requests=set()
        request=Pending('a',[],16);request.output=[2];request.speculative_prompt=(1,2,3,1)
        verifier=SimpleNamespace(width=4,verify=lambda *args:([9,8,7],2))
        executor=SimpleNamespace(speculative_verifier=verifier,lengths=[5],context=128,kernel_backend='test',
            residual_dtype='float16',model=SimpleNamespace(eos={9},tokenizer=SimpleNamespace(decode=lambda tokens,**kw:''.join(map(str,tokens)))))
        active={0:(request,4)};ready=[(0,request,4)]
        return runtime,request,executor,active,ready

    def test_eos_trims_committed_cache_and_emits_no_later_drafts(self):
        runtime,r,executor,active,ready=self.setup_step()
        self.assertTrue(runtime._speculate(ready,active,executor))
        self.assertEqual(r.output,[2,9]);self.assertEqual(executor.lengths,[6]);self.assertEqual(active,{})
        self.assertEqual(r.events.qsize(),1);self.assertEqual(r.final['finish_reason'],'stop')

    def test_cancelled_emission_does_not_commit_verifier_cache(self):
        runtime,r,executor,active,ready=self.setup_step();r.cancelled.set()
        self.assertTrue(runtime._speculate(ready,active,executor))
        self.assertEqual(r.output,[2]);self.assertEqual(executor.lengths,[5]);self.assertEqual(active,{})

    def test_backpressure_concurrency_and_context_fall_back(self):
        runtime,r,executor,active,ready=self.setup_step()
        for _ in range(7):r.events.put_nowait({})
        self.assertFalse(runtime._speculate(ready,active,executor))
        while not r.events.empty():r.events.get_nowait()
        active[1]=(Pending('a',[],16),4)
        self.assertFalse(runtime._speculate(ready,active,executor));del active[1]
        executor.lengths[0]=126;self.assertFalse(runtime._speculate(ready,active,executor))
        self.assertEqual(r.output,[2])

    def test_partial_queue_capacity_bounds_the_proposal(self):
        runtime,r,executor,active,ready=self.setup_step()
        for _ in range(5):r.events.put_nowait({})
        def verify(current,drafts,start,slot):
            self.assertLessEqual(len(drafts),2)
            return [3]*(len(drafts)+1),len(drafts)
        executor.speculative_verifier.verify=verify
        self.assertTrue(runtime._speculate(ready,active,executor))
        self.assertEqual(r.events.qsize(),8)
        self.assertEqual(executor.lengths,[8])

    def test_verifier_rejects_invalid_used_predictions_before_acceptance(self):
        import torch
        from unittest.mock import Mock
        from damendaemon.speculative import Verifier
        from damendaemon.sampling import greedy_tokens
        verifier=Verifier.__new__(Verifier)
        verifier.width=4;verifier.context=32;verifier.slots=2
        verifier.copied=Mock();verifier.graph=Mock();verifier.metadata=Mock()
        verifier.staging=object();verifier.staging_values=[0]*12
        logits=torch.zeros(4,5)
        logits[0,2]=1;logits[1,3]=1;logits[2,4]=1
        for index in range(3):
            for invalid in (float('nan'),float('inf'),-float('inf')):
                broken=logits.clone();broken[index,:]=invalid
                verifier.next_ids=greedy_tokens(broken)
                with self.subTest(index=index,invalid=invalid),self.assertRaisesRegex(RuntimeError,'non-finite speculative'):
                    verifier.verify(1,[2,3],0,0)
        # The fourth row is graph padding and never contributes a token.
        logits[3,:]=float('nan');verifier.next_ids=greedy_tokens(logits)
        self.assertEqual(verifier.verify(1,[2,3],0,0),([2,3,4],2))
