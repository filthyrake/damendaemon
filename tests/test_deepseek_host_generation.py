from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
import torch
from unittest.mock import patch
from dllm.deepseek_host_generation import HostDeepSeekGenerator,native_session_identity
from dllm.deepseek_host_pipeline_build import build_deepseek_host_pipeline_worker
from dllm.deepseek_generation import TokenStream
from dllm.deepseek_batch_generation import BatchReferenceGenerator
from dllm.deepseek_reference_loader import load_reference_model
from dllm.deepseek_text import DeepSeekText,END_THINK,EOS
from dllm.host_topology import HostTopology
from dllm.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests import test_glm_host_generation as glm,test_serve_cpu_reference as serving
from tests.test_deepseek_batch_tools import completion

TOOLS=[dict(type='function',function=dict(name='lookup',strict=True,parameters=dict(type='object',properties={'value':{'type':'string'}},required=['value'],additionalProperties=False)))]


class DeepSeekHostGenerationTests(unittest.TestCase):
    def links(self,plan):
        links=glm.GlmHostGenerationTests.links(self,plan)
        # Full DSML prompts execute real compressed-attention CPU arithmetic.
        # A two-second transport-test budget also includes that computation;
        # it is not a latency acceptance gate for this numerical fixture.
        for group in links:
            for transport in group:transport.timeout=10
        return links
    together=glm.GlmHostGenerationTests.together
    pump=glm.GlmHostGenerationTests.pump
    rows=staticmethod(glm.GlmHostGenerationTests.rows)

    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def session(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        path=serving.ReferenceCommandTests.snapshot(self,'deepseek')
        config=json.loads((path/'config.json').read_text());text=DeepSeekText(path,vocab_size=config['vocab_size'])
        plan=HostTopology(config,[(0,1),(1,3)],b'd'*32,quantization='fp4_fp8_reference',context=4096,context_limit=4096,
            slots=2,chunk_size=2048,frame_bytes=524288,mapped_vocab_size=text.vocab_size,session_identity=native_session_identity(text))
        workers=[build_deepseek_host_pipeline_worker(path,plan,i,group,epoch=1) for i,group in enumerate(self.links(plan))]
        for worker in workers:self.addCleanup(worker.close)
        self.together([worker.handshake for worker in workers]);_,threads=self.pump(workers)
        self.addCleanup(lambda:[thread.join(3) for thread in threads]);self.addCleanup(workers[0].close)
        return text,workers,BatchReferenceGenerator(load_reference_model(path),text,max_context=4096)

    def scripted(self,worker,text,wire=None,ragged=False):
        tokens=text.tokenizer.encode(completion() if wire is None else wire,add_special_tokens=False).ids
        original=worker.stage.output_logits;step=[0]
        def choose(hidden):
            logits=original(hidden);index=max(0,step[0]-1) if ragged else step[0]
            logits.zero_();logits[:,tokens[index]]=10;step[0]+=1;return logits
        return tokens,step,patch.object(worker.stage,'output_logits',choose)

    def test_single_equal_ragged_thinking_rows_match_local_control_and_reuse(self):
        text,workers,control=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=8);self.addCleanup(runner.close)
        message=[dict(role='user',content='Hi')]
        for rows,limits,ragged in (([message],3,False),([message]*2,[2,3],False),([message,[dict(role='user',content='Longer request')]],[1,3],True)):
            actual=list(runner.generate_batch(rows,max_new_tokens=limits,ragged=ragged));expected=list(control.generate_batch(rows,max_new_tokens=limits,ragged=ragged))
            self.assertEqual(self.rows(actual),self.rows(expected));self.assertTrue(all(w.stage.states==[None,None] for w in workers))
        for thinking in (False,True):
            self.assertEqual(list(runner.generate(message,max_new_tokens=2,enable_thinking=thinking)),list(control.generate(message,max_new_tokens=2,enable_thinking=thinking)))

    def test_policy_rejection_before_claim_and_detached_tokenizer(self):
        text,workers,_=self.session();worker=workers[0];worker.model_identity_verified=True
        with self.assertRaises(ValueError):HostDeepSeekGenerator(worker,text)
        self.assertIsNone(getattr(worker,'_completion_owner',None));worker.model_identity_verified=False
        original=text.eos;text.eos=set()
        with self.assertRaises(ValueError):HostDeepSeekGenerator(worker,text)
        text.eos=original;runner=HostDeepSeekGenerator(worker,text);self.addCleanup(runner.close)
        text.eos=set();text.tokenizer.enable_truncation(1)
        self.assertEqual(runner.text.eos,original);self.assertGreater(len(runner.text.chat_tokens([dict(role='user',content='Hi')])),1)
        with self.assertRaises(ValueError):HostDeepSeekGenerator(worker,runner.text)
        from dllm.reference_serving import ReferenceRuntime
        with self.assertRaises(ValueError):ReferenceRuntime(runner)

    def test_invalid_requests_leave_healthy_epoch_without_dispatch(self):
        text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text);self.addCleanup(runner.close);message=[dict(role='user',content='Hi')]
        requests=[lambda:runner.generate([],max_new_tokens=1),lambda:runner.generate(message,max_new_tokens=True),
            lambda:runner.generate_batch([message]*3,max_new_tokens=1),lambda:runner.generate(message,max_new_tokens=4096),
            lambda:runner.generate_batch([message,[dict(role='user',content='Longer')]],max_new_tokens=1),
            lambda:runner.generate(message,max_new_tokens=1,tools=[]),lambda:runner.generate(message,max_new_tokens=1,enable_thinking=1)]
        with patch.object(workers[0],'prefill_chunk',side_effect=AssertionError('dispatch')):
            for request in requests:
                with self.assertRaises(ValueError):list(request())
        self.assertFalse(runner.completion.stopped.is_set());self.assertTrue(list(runner.generate(message,max_new_tokens=1)))

    def test_dsml_strict_tools_and_ragged_rows_after_real_arithmetic(self):
        for ragged in (False,True):
            text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
            tokens,_,select=self.scripted(workers[-1],text,ragged=ragged)
            rows=[[dict(role='user',content='Hi')],[dict(role='user',content='Longer request')]]
            with select:
                events=list(runner.generate_batch_tools(rows,TOOLS,max_new_tokens=len(tokens),ragged=True)) if ragged else list(runner.generate_tools(rows[0],TOOLS,max_new_tokens=len(tokens)))
            for row,pieces in self.rows(events).items():
                self.assertEqual(pieces[-1]['finish_reason'],'tool_calls');self.assertEqual(pieces[-1]['usage']['prompt_tokens'],len(text.chat_tokens(rows[row],tools=TOOLS)))
                call=next(e['tool_calls'] for e in pieces if 'tool_calls' in e)[0]
                self.assertEqual(call['function']['name'],'lookup');self.assertEqual(json.loads(call['function']['arguments']),{'value':'first'})
            self.assertTrue(all(w.stage.states==[None,None] for w in workers))

    def test_tool_contract_is_detached_and_bad_call_discards_epoch_without_success(self):
        for bad in (None,'unknown','schema','truncated'):
            text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
            tools=deepcopy(TOOLS)
            wire=completion(name='unknown') if bad=='unknown' else completion(value=5) if bad=='schema' else completion()[:-len(EOS)-10] if bad=='truncated' else None
            tokens,_,select=self.scripted(workers[-1],text,wire)
            original=workers[0].prefill_chunk
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs);tools[0]['function']['name']='changed';tools[0]['function']['parameters']['properties']['value']['type']='integer';return result
            events=[]
            with select,patch.object(workers[0],'prefill_chunk',mutate):
                if bad:
                    with self.assertRaises(ValueError):events.extend(runner.generate_tools([dict(role='user',content='Hi')],tools,max_new_tokens=len(tokens)))
                    self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events));self.assertTrue(runner.completion.stopped.is_set())
                else:
                    events=list(runner.generate_tools([dict(role='user',content='Hi')],tools,max_new_tokens=len(tokens)))
                    self.assertEqual(events[-1]['finish_reason'],'tool_calls')

    def test_abandonment_cancellation_and_remote_loss_discard_owned_session(self):
        for failure in ('abandon','cancel','remote'):
            text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text);self.addCleanup(runner.close)
            stream=runner.generate([dict(role='user',content='Hi')],max_new_tokens=8);next(stream)
            if failure=='abandon':stream.close()
            elif failure=='cancel':
                with cancellation_scope(lambda:True):
                    with self.assertRaises(ReferenceCancelled):next(stream)
            else:
                workers[1].close()
                with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):list(stream)
            self.assertTrue(runner.completion.stopped.is_set());self.assertTrue(workers[0].closed)

    def test_buffered_tool_cancellation_and_finish_close_suppress_call_publication(self):
        for failure in ('cancel','finish_close'):
            text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
            tokens,step,select=self.scripted(workers[-1],text);events=[];finish=TokenStream.finish
            def close_on_finish(parser,**options):
                result=finish(parser,**options);runner.close();return result
            with select:
                if failure=='cancel':
                    with cancellation_scope(lambda:step[0]>=2):
                        with self.assertRaises(ReferenceCancelled):events.extend(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=len(tokens)))
                else:
                    with patch.object(TokenStream,'finish',close_on_finish):
                        with self.assertRaises(RuntimeError):events.extend(runner.generate_tools([dict(role='user',content='Hi')],TOOLS,max_new_tokens=len(tokens)))
            self.assertFalse(any('tool_calls' in e or 'finish_reason' in e for e in events));self.assertTrue(runner.completion.stopped.is_set())

    def test_thinking_transition_after_real_host_execution(self):
        text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text,chunk_size=2048);self.addCleanup(runner.close)
        tokens,_,select=self.scripted(workers[-1],text,'reason'+END_THINK+'answer'+EOS)
        with select:events=list(runner.generate([dict(role='user',content='Hi')],enable_thinking=True,max_new_tokens=len(tokens)))
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events),'reason')
        self.assertEqual(''.join(e.get('content','') for e in events),'answer');self.assertEqual(events[-1]['finish_reason'],'stop')

    def test_failed_parser_cleanup_retains_owner_and_suppresses_final(self):
        text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text);self.addCleanup(runner.close);events=[]
        with patch.object(TokenStream,'close',side_effect=RuntimeError('parser close')):
            with self.assertRaisesRegex(RuntimeError,'parser close'):events.extend(runner.generate([dict(role='user',content='Hi')],max_new_tokens=1))
            self.assertTrue(runner.failed_cleanups)
            with self.assertRaisesRegex(RuntimeError,'parser close'):runner.cleanup()
        self.assertFalse(any('finish_reason' in e for e in events));self.assertTrue(runner.completion.stopped.is_set());runner.cleanup();self.assertFalse(runner.failed_cleanups)

    def test_partial_parser_construction_closes_created_parser_before_dispatch(self):
        text,workers,_=self.session();runner=HostDeepSeekGenerator(workers[0],text);self.addCleanup(runner.close)
        original=runner._stream;created=[]
        def make(codec):
            if created:raise RuntimeError('second parser')
            stream=original(codec);created.append(stream);return stream
        with patch.object(runner,'_stream',make),patch.object(TokenStream,'close',autospec=True) as close,patch.object(workers[0],'prefill_chunk',side_effect=AssertionError('dispatch')):
            with self.assertRaisesRegex(RuntimeError,'second parser'):list(runner.generate_batch([[dict(role='user',content='Hi')]]*2,max_new_tokens=1))
            close.assert_called_once_with(created[0])
        self.assertFalse(runner.completion.stopped.is_set());self.assertTrue(list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=1)))
