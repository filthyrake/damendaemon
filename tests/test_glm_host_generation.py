import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dllm.glm_host_generation import HostGlmGenerator,native_session_identity
from dllm.glm_host_pipeline_build import build_glm_host_pipeline_worker
from dllm.host_topology import HostTopology
from dllm.glm_tool_text import GlmToolText
from dllm.glm_tool_output import GlmToolTokenStream
from dllm.glm_batch_generation import BatchReferenceGenerator
from dllm.host_tensor_channel import HostTensorChannel
from tests import test_host_tls
from dllm.glm_text import END_THINK,EOS
from dllm.glm_flash_tools import FlashTools
from dllm.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests import test_cpu_host_pipeline_build as build_tests,test_host_pipeline_worker as worker_tests
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture,floating_control
from tests.test_glm_tools import TOOLS,CALLS


class GlmHostGenerationTests(unittest.TestCase):
    def links(self,plan):
        fixture_tls=test_host_tls.HostTlsTests();self.addCleanup(fixture_tls.doCleanups)
        links=[]
        for rank,sock in enumerate(fixture_tls.pair()):
            transport=HostTensorChannel(sock,plan.stage_identity(1),rank,max_frame_bytes=plan.frame_bytes,timeout=2)
            self.addCleanup(transport.close);links.append([transport])
        return links
    together=build_tests.CpuHostPipelineBuildTests.together
    pump=worker_tests.HostPipelineWorkerTests.pump

    def session(self,**policy):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)
        base=text_fixture(path);text=GlmToolText(path,vocab_size=base.vocab_size)
        config,_,values=fixture(path,vocab_size=text.vocab_size)
        plan=HostTopology(config,[(0,2),(2,3)],b'g'*32,quantization='fp8_reference',
            context=4096,context_limit=4096,slots=2,chunk_size=2048,frame_bytes=131072,
            mapped_vocab_size=text.tokenizer_size,session_identity=native_session_identity(text),**policy)
        links=self.links(plan)
        workers=[build_glm_host_pipeline_worker(path,plan,i,group,epoch=1) for i,group in enumerate(links)]
        for worker in workers:self.addCleanup(worker.close)
        self.together([worker.handshake for worker in workers]);errors,threads=self.pump(workers)
        self.addCleanup(lambda:[thread.join(3) for thread in threads])
        self.addCleanup(workers[0].close)
        return text,workers,BatchReferenceGenerator(floating_control(config,values),text,max_context=4096)

    @staticmethod
    def rows(events):
        # Host completion emits final events only after every remote release;
        # compare per-row semantic streams independently of batch interleaving.
        result={}
        for event in events:
            event=dict(event);row=event.pop('row',0);result.setdefault(row,[]).append(event)
        return result

    def test_native_single_ragged_and_equal_rows_match_independent_control_and_reuse(self):
        text,workers,control=self.session();runner=HostGlmGenerator(workers[0],text,chunk_size=8)
        self.addCleanup(runner.close);messages=[dict(role='user',content='Hi')]
        for rows,limits,ragged in (([messages],3,False),([messages,messages],[2,3],False),
                ([messages,[dict(role='user',content='A longer prompt')]], [1,3],True)):
            actual=list(runner.generate_batch(rows,max_new_tokens=limits,ragged=ragged))
            expected=list(control.generate_batch(rows,max_new_tokens=limits,ragged=ragged))
            self.assertEqual(self.rows(actual),self.rows(expected))
            self.assertTrue(all(worker.stage.states==[None,None] for worker in workers))
        self.assertEqual(list(runner.generate(messages,max_new_tokens=2)),list(control.generate(messages,max_new_tokens=2)))

    def test_policy_rejection_precedes_claim_and_tokenizer_is_snapshotted(self):
        text,workers,_=self.session();worker=workers[0]
        worker.model_identity_verified=True
        with self.assertRaises(ValueError):HostGlmGenerator(worker,text)
        self.assertIsNone(getattr(worker,'_completion_owner',None));worker.model_identity_verified=False
        original=text.eos;text.eos=frozenset()
        with self.assertRaises(ValueError):HostGlmGenerator(worker,text)
        text.eos=original
        runner=HostGlmGenerator(worker,text);self.addCleanup(runner.close)
        text.eos=frozenset();text.tokenizer.enable_truncation(1)
        self.assertEqual(runner.text.eos,original)
        self.assertGreater(len(runner.text.chat_tokens([dict(role='user',content='Hello')])),1)
        with self.assertRaises(ValueError):HostGlmGenerator(worker,runner.text)

    def test_invalid_requests_do_not_dispatch_or_discard_healthy_epoch(self):
        text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text);self.addCleanup(runner.close)
        messages=[dict(role='user',content='Hi')]
        requests=[lambda:runner.generate([],max_new_tokens=1),lambda:runner.generate(messages,max_new_tokens=True),
            lambda:runner.generate_batch([messages]*3,max_new_tokens=1),
            lambda:runner.generate_batch([messages,[dict(role='user',content='Longer')]],max_new_tokens=1),
            lambda:runner.generate_tools(messages,[],max_new_tokens=1),
            lambda:runner.generate(messages,max_new_tokens=4096)]
        with patch.object(workers[0],'prefill_chunk',side_effect=AssertionError('dispatch')):
            for request in requests:
                with self.assertRaises(ValueError):list(request())
        self.assertFalse(runner.completion.stopped.is_set())
        self.assertTrue(list(runner.generate(messages,max_new_tokens=1)))

    def scripted(self,worker,text):
        wire=END_THINK+FlashTools(TOOLS).render(CALLS)+EOS
        tokens=text.tokenizer.encode(wire,add_special_tokens=False).ids
        original=worker.stage.output_logits;step=[0]
        def select(hidden):
            logits=original(hidden);logits.zero_();logits[:,tokens[step[0]]]=10;step[0]+=1
            return logits
        return tokens,step,patch.object(worker.stage,'output_logits',select)

    def test_native_tools_after_real_stage_execution_and_cleanup_before_final(self):
        text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
        tokens,step,selection=self.scripted(workers[-1],text)
        with selection:
            events=list(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=len(tokens)))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        calls=next(event['tool_calls'] for event in events if 'tool_calls' in event)
        self.assertEqual(calls[0]['function']['name'],CALLS[0]['function']['name'])
        self.assertEqual(json.loads(calls[0]['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        self.assertTrue(all(worker.stage.states==[None,None] for worker in workers))

    def test_abandonment_and_cancellation_discard_session(self):
        for cancel in (False,True):
            text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text);self.addCleanup(runner.close)
            stream=runner.generate([dict(role='user',content='Hi')],max_new_tokens=8);next(stream)
            if cancel:
                with cancellation_scope(lambda:True):
                    with self.assertRaises(ReferenceCancelled):next(stream)
            else:stream.close()
            self.assertTrue(runner.completion.stopped.is_set());self.assertTrue(workers[0].closed)
            with self.assertRaises(RuntimeError):list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=1))

    def test_parser_close_failure_suppresses_final_retains_cleanup_and_discards_epoch(self):
        text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text);self.addCleanup(runner.close)
        events=[]
        with patch.object(GlmToolTokenStream,'close',side_effect=RuntimeError('parser close')):
            with self.assertRaisesRegex(RuntimeError,'parser close'):
                events.extend(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=1))
            self.assertTrue(runner.failed_cleanups)
            with self.assertRaisesRegex(RuntimeError,'parser close'):runner.cleanup()
        self.assertFalse(any('finish_reason' in event for event in events))
        self.assertTrue(runner.completion.stopped.is_set());runner.cleanup();self.assertFalse(runner.failed_cleanups)

    def test_ragged_native_tools_preserve_row_identity(self):
        text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
        tokens=text.tokenizer.encode(END_THINK+FlashTools(TOOLS).render(CALLS)+EOS,add_special_tokens=False).ids
        original=workers[-1].stage.output_logits;step=0
        def select(hidden):
            nonlocal step
            logits=original(hidden);logits.zero_();logits[:,tokens[max(0,step-1)]]=10;step+=1
            return logits
        rows=[[dict(role='user',content='Hi')],[dict(role='user',content='A longer prompt')]]
        with patch.object(workers[-1].stage,'output_logits',select):
            events=list(runner.generate_batch_tools(rows,TOOLS,max_new_tokens=len(tokens),ragged=True))
        for row,events in self.rows(events).items():
            self.assertEqual(events[-1]['finish_reason'],'tool_calls')
            self.assertEqual(events[-1]['usage']['prompt_tokens'],len(text.tool_chat_tokens(rows[row],TOOLS)))
            self.assertEqual(next(e['tool_calls'] for e in events if 'tool_calls' in e)[0]['function']['name'],CALLS[0]['function']['name'])

    def test_tool_cancellation_and_remote_failure_do_not_report_success(self):
        for failure in ('cancel','remote'):
            text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
            tokens,step,selection=self.scripted(workers[-1],text);events=[]
            with selection:
                if failure=='cancel':
                    with cancellation_scope(lambda:step[0]>=2):
                        with self.assertRaises(ReferenceCancelled):
                            events.extend(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=len(tokens)))
                else:
                    with patch.object(workers[-1].stage,'forward_layers',side_effect=RuntimeError('remote')):
                        with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):
                            events.extend(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=len(tokens)))
            self.assertFalse(any('finish_reason' in e or 'tool_calls' in e for e in events))
            self.assertTrue(runner.completion.stopped.is_set());self.assertTrue(workers[0].closed)

    def test_external_close_during_parser_finish_suppresses_success(self):
        text,workers,_=self.session();runner=HostGlmGenerator(workers[0],text);self.addCleanup(runner.close)
        original=GlmToolTokenStream.finish
        def finish(parser,**options):
            result=original(parser,**options);runner.close();return result
        with patch.object(GlmToolTokenStream,'finish',finish):
            with self.assertRaises(RuntimeError):list(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=1))
