from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch
import torch
from dllm.host_completion import HostCompletion
from dllm.hybrid_pipeline import PipelineCancelled
from tests import test_glm_flash_host_embeddings as fixtures


class FlashEmbeddingCompletionTests(unittest.TestCase):
    setUp=fixtures.FlashHostEmbeddingTests.setUp
    plan=fixtures.FlashHostEmbeddingTests.plan
    session=fixtures.FlashHostEmbeddingTests.session
    pump=fixtures.FlashHostEmbeddingTests.pump

    def prepared(self,count=2,pump=True):
        _,owners,control=self.session(count);runner=HostCompletion(owners[0])
        if pump:
            errors,threads=self.pump(owners)
            def finish():
                runner.close()
                for thread in threads:thread.join(3);self.assertFalse(thread.is_alive())
            self.addCleanup(finish)
        return runner,owners,control

    def expected(self,control,rows,limits,eos=()):
        tokens=[];reasons=[]
        for row,limit in zip(rows,limits):
            logits,state=control.prefill_embeddings(row[None],capacity=16)
            try:
                result=[];reason='length'
                for position in range(limit):
                    scores=logits[:,:30].clone();scores[:,29]=-float('inf');token=int(scores.argmax(-1));result.append(token)
                    if token in eos:reason='stop';break
                    if position+1<limit:logits=control.forward(torch.tensor([[token]]),state,position=len(row)+position)
                tokens.append(result);reasons.append(reason)
            finally:state.close()
        events=[]
        for step in range(max(map(len,tokens))):
            for row,result in enumerate(tokens):
                if step<len(result):events.append(dict(row=row,token_id=result[step]))
        for row,result in enumerate(tokens):
            events.append(dict(row=row,finish_reason=reasons[row],usage=dict(prompt_tokens=len(rows[row]),completion_tokens=len(result),total_tokens=len(rows[row])+len(result))))
        return events

    def test_real_two_three_stage_equal_ragged_chunked_controls_and_reuse(self):
        for count in (2,3):
            runner,owners,control=self.prepared(count)
            for lengths,limits in (([3,3],[2,2]),([4,1],[1,4])):
                rows=[torch.randn(length,16)*.2 for length in lengths]
                expected=self.expected(control,rows,limits)
                self.assertEqual(list(runner.generate_embedding_batch(rows,max_new_tokens=limits,chunk_size=2)),expected)
                self.assertTrue(all(owner.channel.state.lengths==[0,0] for owner in owners))
                self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))
            row=torch.randn(2,16)
            expected=[{k:v for k,v in event.items() if k!='row'} for event in self.expected(control,[row],[3])]
            self.assertEqual(list(runner.generate_embeddings(row,max_new_tokens=3,chunk_size=1)),expected)
            # The same owner can still service ordinary token requests.
            self.assertEqual(list(runner.generate([1],max_new_tokens=1))[-1]['finish_reason'],'length')
            runner.close()

    def test_malformed_preflight_preserves_epoch_and_does_not_dispatch(self):
        runner,owners,_=self.prepared();first=owners[0];row=torch.zeros(2,16)
        cases=[([row],dict(max_new_tokens=0)),([row],dict(max_new_tokens=True)),([row],dict(max_new_tokens=[1,2])),
            ([row]*3,{}),([],{}),([row.double()],{}),([row[:0]],{}),([row[:,:15]],{}),([row[:,None]],{}),
            ([torch.full((2,16),float('inf'))],{}),([torch.empty(2,16,device='meta')],{}),
            ([row],dict(max_new_tokens=15)),([row],dict(chunk_size=True)),([torch.zeros(9,16)],dict(chunk_size=9)),([row],dict(cancelled=True))]
        with patch.object(first,'prefill_embeddings',side_effect=AssertionError('invalid dispatch')):
            for rows,options in cases:
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):list(runner.generate_embedding_batch(rows,**(dict(max_new_tokens=1)|options)))
            with patch.object(first.stage.stage,'max_output_bytes',200):
                with self.assertRaises(ValueError):list(runner.generate_embedding_batch([row,row],max_new_tokens=1,chunk_size=1))
            with self.assertRaises(ValueError):list(runner.generate_embeddings(row,max_new_tokens=[1]))
        self.assertEqual(first.channel.state.sequence,1);self.assertTrue(first.ready)
        self.assertEqual(list(runner.generate_embeddings(row,max_new_tokens=1))[-1]['finish_reason'],'length')

    def test_detaches_future_chunks_and_other_rows_before_first_dispatch(self):
        runner,owners,control=self.prepared();rows=[torch.randn(4,16),torch.randn(3,16)]
        expected=self.expected(control,[row.clone() for row in rows],[2,3]);limits=[2,3]
        original=owners[0].prefill_embeddings
        def prefill(*args):
            for row in rows:row.zero_()
            limits[:]=[1,1]
            return original(*args)
        with patch.object(owners[0],'prefill_embeddings',prefill):
            self.assertEqual(list(runner.generate_embedding_batch(rows,max_new_tokens=limits,chunk_size=2)),expected)

    def test_eos_release_before_final_and_release_failure_has_no_success(self):
        runner,owners,_=self.prepared();row=torch.randn(3,16)
        token=list(runner.generate_embeddings(row,max_new_tokens=1))[0]['token_id'];runner.eos_ids=frozenset([token])
        with patch.object(owners[0],'decode',side_effect=AssertionError('decode after EOS')):
            events=list(runner.generate_embeddings(row,max_new_tokens=4,chunk_size=2))
        self.assertEqual(events[-1]['finish_reason'],'stop');self.assertEqual(events[-1]['usage']['completion_tokens'],1)
        self.assertTrue(all(not any(owner.channel.state.lengths) for owner in owners))
        emitted=[]
        with patch.object(owners[0],'release',side_effect=RuntimeError('release failed')):
            with self.assertRaisesRegex(RuntimeError,'release failed'):
                for event in runner.generate_embeddings(row,max_new_tokens=1):emitted.append(event)
        self.assertFalse(any('finish_reason' in event for event in emitted));self.assertTrue(runner.stopped.is_set())

    def test_cancel_after_chunk_retires_epoch_and_cleanup_can_retry(self):
        runner,owners,_=self.prepared();cancel=threading.Event();original=owners[0].prefill_embeddings
        def prefill(*args):
            value=original(*args);cancel.set();return value
        with patch.object(owners[0],'prefill_embeddings',prefill),patch.object(owners[0].stage,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaises(PipelineCancelled) as caught:
                list(runner.generate_embeddings(torch.randn(4,16),max_new_tokens=2,chunk_size=2,cancelled=cancel.is_set))
        self.assertTrue(caught.exception.__notes__);self.assertTrue(runner.stopped.is_set());self.assertFalse(owners[0].closed)
        runner.cleanup();self.assertTrue(owners[0].closed)

    def test_abandonment_and_concurrent_request_preserve_owner_contract(self):
        runner,owners,_=self.prepared();row=torch.randn(2,16)
        stream=runner.generate_embeddings(row,max_new_tokens=3);next(stream)
        with self.assertRaisesRegex(RuntimeError,'active'):list(runner.generate_embeddings(row,max_new_tokens=1))
        self.assertTrue(owners[0].ready);self.assertEqual(list(stream)[-1]['usage']['completion_tokens'],3)
        stream=runner.generate_embeddings(row,max_new_tokens=3);next(stream);stream.close()
        self.assertTrue(runner.stopped.is_set());self.assertTrue(owners[0].closed)
        with self.assertRaises(PipelineCancelled):list(runner.generate_embeddings(row,max_new_tokens=1))

    def test_close_interrupts_blocked_prepared_prefill(self):
        runner,owners,_=self.prepared(pump=False);entered=threading.Event();original=owners[0].prefill_embeddings
        def prefill(*args):entered.set();return original(*args)
        with patch.object(owners[0],'prefill_embeddings',prefill),ThreadPoolExecutor(1) as pool:
            future=pool.submit(lambda:list(runner.generate_embeddings(torch.zeros(2,16),max_new_tokens=1)))
            self.assertTrue(entered.wait(2));runner.close()
            with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):future.result(3)
        self.assertTrue(owners[0].closed)

    def test_plain_token_session_rejects_features_without_poisoning(self):
        from tests.test_glm_flash_host_pipeline import FlashHostPipelineTests
        with patch.object(self,'plan',side_effect=lambda **kw:FlashHostPipelineTests.plan(self,**kw)):
            runner,owners,_=self.prepared()
        with self.assertRaises(ValueError):list(runner.generate_embeddings(torch.zeros(2,16),max_new_tokens=1))
        self.assertTrue(owners[0].ready)
        self.assertEqual(list(runner.generate([1],max_new_tokens=1))[-1]['finish_reason'],'length')
