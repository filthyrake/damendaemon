import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from damendaemon.reference_serving import ReferenceRuntime,handler_for_reference
from damendaemon.runtime import BusyError
from damendaemon.server import BoundedHTTPServer
from damendaemon.http_auth import AuthPolicy
from damendaemon.glm_tool_text import GlmToolText
from damendaemon.glm_tool_generation import GlmToolReferenceGenerator
from damendaemon.glm_flash_tool_text import FlashToolText
from damendaemon.glm_flash_tool_generation import FlashToolReferenceGenerator
from damendaemon.deepseek_generation import ReferenceGenerator as DeepSeekGenerator
from damendaemon.glm_flash_tools import FlashTools
from damendaemon.deepseek_tools import render_tool_calls
from damendaemon.glm_text import END_THINK,EOS
from damendaemon import deepseek_text
from tests import test_glm_generation as glm
from tests import test_glm_flash_generation as flash
from tests import test_deepseek_generation as deepseek
from tests.test_glm_reference_loader import fixture
from damendaemon.glm_reference_loader import load_reference_model

TOOLS=[dict(type='function',function=dict(name='lookup',strict=True,parameters={
    'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}))]
CALLS=[dict(type='function',function=dict(name='lookup',arguments='{"value":"false"}'))]


class ReferenceServingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.messages=[dict(role='user',content='Hi')]

    def runtime(self,family,*,tools=False,output=None):
        path=self.path/family;path.mkdir(exist_ok=True)
        if family=='deepseek':
            text=deepseek.text_fixture(path)
            end,eos=deepseek_text.END_THINK,deepseek_text.EOS
            wire='\n\n'+render_tool_calls(CALLS) if tools else ''
            model=deepseek.ScriptedModel(text,output or 'reason'+end+'answer'+wire+eos)
            runner=DeepSeekGenerator(model,text,max_context=4096)
        else:
            base=(glm if family=='glm' else flash).text_fixture(path)
            text=(GlmToolText if family=='glm' else FlashToolText)(path,vocab_size=base.vocab_size)
            wire=FlashTools(TOOLS).render(CALLS) if tools else ''
            model=(glm.ScriptedModel if family=='glm' else flash.ScriptedFlash)(text,output or 'reason'+END_THINK+'answer'+wire+EOS)
            runner=(GlmToolReferenceGenerator if family=='glm' else FlashToolReferenceGenerator)(model,text,max_context=4096)
        runtime=ReferenceRuntime(runner);self.addCleanup(runtime.close)
        return runtime,model

    def serve(self,runtime):
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for_reference(runtime,AuthPolicy()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def close():server.shutdown();server.server_close();thread.join(3)
        self.addCleanup(close)
        return f'http://127.0.0.1:{server.server_port}'

    def body(self,family,**changes):
        result=dict(model='model',messages=self.messages,max_tokens=512)
        if family=='deepseek':result['chat_template_kwargs']={'enable_thinking':True}
        return result|changes

    def post(self,base,body):
        return urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'}),timeout=4)

    def test_rejects_malformed_runner_terminal_sequences_and_releases_owner(self):
        runtime,_=self.runtime('glm')
        final=dict(finish_reason='stop',usage=dict(prompt_tokens=1,completion_tokens=1))
        cases=[[],[final,final],[final,dict(content='late')],['not an event'],
               [dict(final,finish_reason='')],[dict(final,usage=[])],
               [dict(finish_reason='stop',usage=dict(prompt_tokens=1))],
               [dict(final,usage=dict(prompt_tokens=True,completion_tokens=1))],
               [dict(final,usage=dict(prompt_tokens=1,completion_tokens=-1))]]
        for events in cases:
            with self.subTest(events=events):
                closed=threading.Event()
                def stream(*args,**kwargs):
                    try:yield from events
                    finally:closed.set()
                with patch.object(runtime.runner,'generate',side_effect=stream):
                    with self.assertRaises(RuntimeError):
                        list(runtime.generate('model',self.messages,2))
                self.assertTrue(closed.is_set())
                self.assertFalse(runtime.active)
                self.assertFalse(runtime.lock.locked())
                self.assertTrue(runtime.is_healthy())
        # The rejected iterator must not poison the next request.
        self.assertEqual(list(runtime.generate('model',self.messages,512))[-1]['finish_reason'],'stop')

    def test_http_all_families_reasoning_content_and_native_strict_tools(self):
        for family in ('glm','flash','deepseek'):
            runtime,model=self.runtime(family,tools=True);base=self.serve(runtime)
            ids=[]
            for _ in range(2):
                with self.post(base,self.body(family,tools=TOOLS)) as response:result=json.load(response)
                choice=result['choices'][0];message=choice['message']
                self.assertEqual(choice['finish_reason'],'tool_calls');self.assertEqual(message['reasoning_content'],'reason')
                self.assertEqual(message['content'].strip(),'answer')
                call=message['tool_calls'][0];ids.append(call['id'])
                self.assertEqual(json.loads(call['function']['arguments']),{'value':'false'})
                self.assertEqual(result['usage']['total_tokens'],result['usage']['prompt_tokens']+result['usage']['completion_tokens'])
                self.assertEqual(result['damendaemon_timing']['source_tier'],'cpu_reference')
                self.assertFalse(result['damendaemon_timing']['official_model_qualified']);self.assertTrue(model.states[-1].closed)
            self.assertNotEqual(*ids)
            with urlopen(base+'/v1/models/status') as response:self.assertEqual(json.load(response)['device'],'cpu')

    def test_native_tool_sse_and_history_continuation(self):
        for family in ('glm','flash','deepseek'):
            runtime,model=self.runtime(family,tools=True);base=self.serve(runtime)
            with self.post(base,self.body(family,tools=TOOLS,stream=True,stream_options={'include_usage':True})) as response:
                wire=response.read().decode()
            events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
            calls=[call for e in events for choice in e['choices'] for call in choice.get('delta',{}).get('tool_calls',[])]
            self.assertEqual(len(calls),1);self.assertEqual(calls[0].pop('index'),0)
            self.assertIn('data: [DONE]',wire);self.assertTrue(events[-1]['usage'])
            history=self.messages+[dict(role='assistant',content=None,tool_calls=calls),
                dict(role='tool',tool_call_id=calls[0]['id'],content='found')]
            with self.post(base,self.body(family,tools=TOOLS,messages=history)) as response:result=json.load(response)
            self.assertEqual(result['choices'][0]['finish_reason'],'tool_calls')
            self.assertNotEqual(result['choices'][0]['message']['tool_calls'][0]['id'],calls[0]['id'])

    def test_bad_options_and_histories_before_prefill(self):
        for family in ('glm','flash','deepseek'):
            runtime,model=self.runtime(family);base=self.serve(runtime)
            changes=[dict(temperature=1),dict(stop='stop'),dict(response_format={'type':'json_object'}),
                     dict(tools=TOOLS,tool_choice='required'),dict(tools=TOOLS,tool_choice={'type':'function','function':{'name':'lookup'}}),
                     dict(logprobs=True),dict(reasoning_effort='medium'),dict(max_tokens=4097),
                     dict(tools=TOOLS,messages=self.messages+[dict(role='tool',tool_call_id='foreign',content='bad')])]
            with patch.object(model,'prefill',side_effect=AssertionError('prefill')):
                for change in changes:
                    with self.assertRaises(HTTPError) as error:self.post(base,self.body(family,**change))
                    self.assertEqual(error.exception.code,400)
            self.assertFalse(model.states)

    def test_busy_cancel_shutdown_and_cleanup_retry(self):
        runtime,model=self.runtime('glm');runner=runtime.runner
        stream=runtime.generate('model',self.messages,512);next(stream)
        with self.assertRaises(BusyError):next(runtime.generate('model',self.messages,512))
        stream.close();self.assertTrue(model.states[-1].closed)
        stream=runtime.generate('model',self.messages,512);next(stream)
        with self.assertRaises(BusyError):runtime.close()
        with self.assertRaises(BusyError):list(stream)
        self.assertTrue(runtime.closed);self.assertTrue(model.states[-1].closed)
        runtime,model=self.runtime('flash');stream=runtime.generate('model',self.messages,512);next(stream)
        state=model.states[-1]
        with patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            stream.close()
            self.assertTrue(runtime.manage('status')['cleanup_pending'])
            with self.assertRaisesRegex(RuntimeError,'cleanup'):runtime.close()
            self.assertIsNotNone(runtime.runner)
        runtime.close();self.assertIsNone(runtime.runner);self.assertTrue(state.closed)
        with self.assertRaises(ValueError):ReferenceRuntime(runner)

    def test_actual_packed_glm_http_and_closed_health(self):
        base_text=glm.text_fixture(self.path);fixture(self.path,vocab_size=base_text.vocab_size)
        text=GlmToolText(self.path,vocab_size=base_text.vocab_size);model=load_reference_model(self.path);model.head.zero_()
        runner=GlmToolReferenceGenerator(model,text,max_context=128)
        expected=list(runner.generate(self.messages,max_new_tokens=2))
        runtime=ReferenceRuntime(runner);self.addCleanup(runtime.close);base=self.serve(runtime)
        with self.post(base,dict(model='model',messages=self.messages,max_tokens=2)) as response:result=json.load(response)
        self.assertEqual(result['choices'][0]['message']['reasoning_content'],''.join(e.get('reasoning_content','') for e in expected))
        runtime.close()
        with self.assertRaises(HTTPError) as error:urlopen(base+'/health')
        self.assertEqual(error.exception.code,503)
