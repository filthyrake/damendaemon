from http.server import ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.bench import events
from dllm.server import handler_for
from dllm.runtime import BusyError


class FakeRuntime:
    models = {"a": "unused"}
    closed = 0

    def model_ids(self):
        return list(self.models)

    def manage(self, action, **options):
        from dllm.management import command
        self.last_management=command(action,**options)
        return {'models': [{'id':'a','residency':'disk'}]}

    def embed(self, alias, inputs, encoding_format='float', dimensions=None):
        from dllm.embeddings import options
        values,encoding_format,dimensions=options(inputs,encoding_format,dimensions)
        self.last_embedding=(alias,values,encoding_format,dimensions)
        return {'object':'list','data':[{'object':'embedding','index':i,'embedding':[0.,1.]} for i in range(len(values))]}

    def rerank(self,alias,query,documents,top_n=None,return_documents=False):
        from dllm.reranking import options,ranked
        query,documents,top_n,return_documents=options(query,documents,top_n,return_documents)
        if alias!='a':raise ValueError('unknown model')
        return {'results':ranked([float(i) for i in range(len(documents))],documents,top_n,return_documents)}

    def score(self,alias,prompt,candidate_token_ids):
        if alias!='a':raise ValueError('unknown model')
        self.last_scoring=(alias,prompt,candidate_token_ids)
        return {'object':'token_scores','normalization':'candidate_set','calibrated':False}

    def generate(self, alias, messages, max_tokens, **kwargs):
        self.last_options = kwargs
        self.last_max_tokens = max_tokens
        try:
            if alias == "busy":
                raise BusyError("request queue is full")
            if alias != "a":
                raise ValueError("unknown model")
            parts=[('Hello',b'Hello'),(' world',b' world')]
            finish='length'
            if messages and messages[0]['content']=='tools':
                fragments=['<tool','_call>\n{"name":"weather","arguments":',
                    '{"city":"Paris","literal":"</tool_call>"}}\n</tool','_call>']
                parts=[(part,part.encode()) for part in fragments];finish='stop'
            if messages and messages[0]['content']=='forced':
                parts=[(part,part.encode()) for part in ['{"name":"weather",','"arguments":{}}']]
                finish='stop'
            if messages and messages[0]['content']=='unicode':
                parts=[('',bytes([240])),('',bytes([159])),('',bytes([166])),('🦙',bytes([153]))]
            harmony=messages and messages[0]['content']=='harmony tools'
            if harmony:
                literal='<tool_call>{"name":"weather","arguments":{}}</tool_call>'
                parts=[(literal,literal.encode()),('',b'')];finish='tool_calls'
            for token,(text,raw) in enumerate(parts,1):
                event={'text':text,'token':token}
                if messages and messages[0]['content']=='reasoning' and token==1:
                    event['reasoning_content']='Separate analysis. '
                if harmony:
                    event['tool_call_format']='harmony'
                    if token==2:event['tool_calls']=[{'id':'call_test','type':'function','function':{'name':'weather','arguments':'{"city":"Paris"}'}}]
                if kwargs.get('logprobs'):
                    event['logprobs']={'token':raw.decode(errors='replace'),'logprob':-.25,'bytes':list(raw),'top_logprobs':[]}
                yield event
            yield {"finish_reason": finish, "usage": {
                "prompt_tokens": 3, "completion_tokens": len(parts), "total_tokens": 3+len(parts)}, "timing": {"queue_seconds": .125}}
        finally:
            self.closed += 1


class ServerTests(unittest.TestCase):
    def test_image_http_validation_requires_opt_in_and_leaves_pixels_to_worker(self):
        messages=[{'role':'user','content':[{'type':'image_url','image_url':
            {'url':'data:image/png;base64,iVBORw0KGgo='}}]}]
        with patch.dict(os.environ,{'DLLM_EXPERIMENTAL_VISION':'0'}):
            with self.assertRaises(HTTPError) as error:self.post(messages=messages)
            self.assertEqual(error.exception.code,400)
        with patch.dict(os.environ,{'DLLM_EXPERIMENTAL_VISION':'1'}), \
                patch('dllm.vision_messages.decode_image_messages') as decode:
            with self.post(messages=messages) as response:
                self.assertEqual(json.load(response)['choices'][0]['message']['content'],'Hello world')
            decode.assert_not_called()
            messages[0]['content'][0]['image_url']['url']='data:image/png;base64,???'
            with self.assertRaises(HTTPError) as error:self.post(messages=messages)
            self.assertEqual(error.exception.code,400)

    @classmethod
    def setUpClass(cls):
        cls.runtime = FakeRuntime()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(cls.runtime))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, **changes):
        data = {"model": "a", "messages": [{"role": "user", "content": "hello"}]}
        data.update(changes)
        return urlopen(Request(self.url + "/v1/chat/completions",
                               data=json.dumps(data).encode(),
                               headers={"Content-Type": "application/json"}), timeout=5)

    def test_nonfinite_model_output_is_an_execution_error(self):
        import torch
        from dllm.sampling import Sampler, Sampling
        from dllm.logprobs import scores
        failures = (
            lambda: Sampler(Sampling()).select(torch.tensor([float('nan'), 1.])),
            lambda: Sampler(Sampling(1)).select(torch.tensor([float('nan'), 1.])),
            lambda: scores(torch.tensor([float('nan'), 1.]), 0, 1),
            lambda: Sampler(Sampling(logit_bias={'0': 1})).select(
                torch.tensor([float('inf'), 1.])),
        )
        for failure in failures:
            def generate(*args, **kwargs):
                failure()
                yield  # Exercise failures during generator iteration.
            with self.subTest(failure=failure), \
                    patch.object(self.runtime, 'generate', side_effect=generate), \
                    patch('dllm.server.traceback.print_exc'):
                with self.assertRaises(HTTPError) as raised:
                    self.post()
                with raised.exception as response:
                    self.assertEqual(response.code, 500)
                    self.assertEqual(json.load(response)['error'], 'model execution failed')
        with self.assertRaises(HTTPError) as raised:
            self.post(temperature=-1)
        with raised.exception as response:
            self.assertEqual(response.code, 400)

    def test_oversized_sampling_integers_return_400_before_dispatch(self):
        with patch.object(self.runtime, 'generate') as generate:
            for name in ('temperature', 'top_p', 'presence_penalty',
                         'frequency_penalty', 'repetition_penalty', 'logit_bias'):
                options = {name: {'0': 10**400} if name == 'logit_bias' else 10**400}
                with self.subTest(name=name), self.assertRaises(HTTPError) as raised:
                    self.post(**options)
                self.assertEqual(raised.exception.code, 400)
                raised.exception.close()
            generate.assert_not_called()

    def test_template_render_failures_return_400_before_inference(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from dllm.qwen2 import ModelWeights
        from dllm.chat_template import compile_template
        model=SimpleNamespace(config={'model_type':'qwen2','vocab_size':4},
            template=compile_template('{{ messages[1].content }}'),
            special_tokens={},tokenizer=Mock())
        def generate(alias,messages,*args,**options):
            ModelWeights.chat_tokens(model,messages)
            raise AssertionError('inference must not be reached')
        for source in ('{{ messages[1].content }}', '{{ messages[0].content + 1 }}',
                       '{{ range(3) | tojson }}', '{{ 1 / (messages | length - 1) }}',
                       '{% for i in range(30) %}{% endfor %}hello'):
            model.template=compile_template(source)
            with patch.object(self.runtime,'generate',side_effect=generate), \
                    patch('dllm.chat_template.MAX_TEMPLATE_ITERATIONS',20):
                for stream in (False,True):
                    with self.subTest(source=source,stream=stream):
                        with self.assertRaises(HTTPError) as raised:
                            self.post(stream=stream)
                        with raised.exception as response:
                            self.assertEqual(response.code,400)
                            self.assertEqual(json.load(response),{'error':'invalid or unsupported request'})
        model.tokenizer.encode.assert_not_called()

    def test_nonstream_completion(self):
        with self.post() as response:
            data = json.load(response)
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(data["usage"]["completion_tokens"], 2)

    def test_stream_usage_preserves_timing_without_extra_events_when_disabled(self):
        with self.post() as response:
            self.assertEqual(json.load(response)['dllm_timing']['queue_seconds'], .125)
        for enabled in (False, True):
            with self.post(stream=True, stream_options={'include_usage': enabled}) as response:
                chunks = [json.loads(value) for value in events(response) if value != '[DONE]']
            timed = [chunk for chunk in chunks if 'dllm_timing' in chunk]
            self.assertEqual(len(timed), int(enabled))
            if enabled:
                self.assertEqual(timed[0]['dllm_timing']['queue_seconds'], .125)
                self.assertIn('usage', timed[0])
                self.assertEqual(timed[0]['choices'], [])

    def test_reasoning_stays_separate_in_stream_and_nonstream(self):
        messages=[{'role':'user','content':'reasoning'}]
        with self.post(messages=messages,reasoning_effort='low') as response:whole=json.load(response)
        self.assertEqual(self.runtime.last_options['reasoning_effort'],'low')
        self.assertEqual(whole['choices'][0]['message'],{'role':'assistant','content':'Hello world',
            'reasoning_content':'Separate analysis. '})
        with self.post(messages=messages,reasoning_effort='low',stream=True) as response:
            chunks=[json.loads(value) for value in events(response) if value!='[DONE]']
        content='';reasoning=''
        for chunk in chunks:
            for choice in chunk.get('choices',[]):
                content+=choice.get('delta',{}).get('content','')
                reasoning+=choice.get('delta',{}).get('reasoning_content','')
        self.assertEqual(content,'Hello world');self.assertEqual(reasoning,'Separate analysis. ')
        for invalid in (True,[],{},'maximum'):
            with self.subTest(invalid=invalid),self.assertRaises(HTTPError) as caught:self.post(reasoning_effort=invalid)
            self.assertEqual(caught.exception.code,400)

    def test_tool_call_stream_and_nonstream_with_result_history(self):
        tools=[{'type':'function','function':{'name':'weather','parameters':{'type':'object'}}}]
        messages=[{'role':'user','content':'tools'}]
        with self.post(messages=messages,tools=tools) as response:whole=json.load(response)
        choice=whole['choices'][0];call=choice['message']['tool_calls'][0]
        self.assertEqual(choice['finish_reason'],'tool_calls')
        self.assertIsNone(choice['message']['content'])
        self.assertEqual(json.loads(call['function']['arguments']),{'city':'Paris','literal':'</tool_call>'})
        with self.post(messages=messages,tools=tools,stream=True,stream_options={'include_usage':True}) as response:
            streamed=list(events(response))
        self.assertEqual(streamed[-1],'[DONE]')
        payloads=[json.loads(value) for value in streamed[:-1]]
        calls=[call for payload in payloads for choice in payload['choices'] for call in choice['delta'].get('tool_calls',[])]
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['function'],call['function'])
        self.assertEqual(calls[0]['index'],0)
        self.assertNotEqual(calls[0]['id'],call['id'])
        self.assertEqual(payloads[-1]['usage'],whole['usage'])
        self.assertEqual(payloads[-2]['choices'][0]['finish_reason'],'tool_calls')
        history=messages+[choice['message'],{'role':'tool','tool_call_id':call['id'],'content':'Sunny'}]
        with self.post(messages=history,tools=tools) as response:self.assertEqual(response.status,200)

    def test_invalid_tool_unicode_returns_400_before_dispatch(self):
        definitions = [
            {'name':'test','description':'\ud800'},
            {'name':'test','parameters':{'type':'object','description':'\udfff'}},
            {'name':'test','parameters':{'type':'object','properties':{'\ud800':{'type':'string'}}}},
            {'name':'test','parameters':{'type':'object','extra':[{'value':'\udc00'}]}},
        ]
        for function in definitions:
            for stream in (False,True):
                with self.subTest(function=function,stream=stream), patch.object(self.runtime,'generate') as generate:
                    with self.assertRaises(HTTPError) as caught:
                        self.post(tools=[{'type':'function','function':function}],stream=stream)
                    with caught.exception as response:self.assertEqual(response.code,400)
                    generate.assert_not_called()

    def test_deep_tool_schema_returns_400_before_dispatch(self):
        nested = json.loads('[' * 600 + '0' + ']' * 600)
        tools = [{'type': 'function', 'function': {'name': 'test',
            'parameters': {'type': 'object', 'extra': nested}}}]
        with patch.object(self.runtime, 'generate') as generate, \
                patch('dllm.server.traceback.print_exc'):
            with self.assertRaises(HTTPError) as caught: self.post(tools=tools)
            with caught.exception as response:
                self.assertEqual(response.code, 400)
            generate.assert_not_called()

    def test_tool_choice_none_and_invalid_combinations(self):
        tools=[{'type':'function','function':{'name':'weather','parameters':{'type':'object'}}}]
        with self.post(messages=[{'role':'user','content':'tools'}],tools=tools,tool_choice='none') as response:
            result=json.load(response)
        self.assertIn('<tool_call>',result['choices'][0]['message']['content'])
        self.assertNotIn('tool_calls',result['choices'][0]['message'])
        self.assertNotIn('tools',self.runtime.last_options)
        for options in ({'tool_choice':'invalid'},{'stop':'stop'},{'logprobs':True},
                {'response_format':{'type':'json_object'}}):
            with self.subTest(options=options),self.assertRaises(HTTPError) as error:
                self.post(tools=tools,**options)
            self.assertEqual(error.exception.code,400)

    def test_chat_logprobs_stream_and_nonstream_match(self):
        with self.post(logprobs=True,top_logprobs=2) as response:
            whole=json.load(response)['choices'][0]['logprobs']['content']
        with self.post(logprobs=True,top_logprobs=2,stream=True) as response:
            chunks=list(events(response))
        scores=[]
        for value in chunks[:-1]:
            for choice in json.loads(value)['choices']:
                if choice.get('logprobs'):scores.extend(choice['logprobs']['content'])
        self.assertEqual(scores,whole)
        self.assertEqual(bytes(b for score in scores for b in score['bytes']).decode(),'Hello world')
        self.assertEqual(self.runtime.last_options['top_logprobs'],2)

    def test_invalid_logprobs_combinations(self):
        for options in ({'logprobs':1},{'top_logprobs':1},{'logprobs':True,'top_logprobs':21},
                        {'logprobs':True,'stop':'x'}):
            with self.subTest(options=options),self.assertRaises(HTTPError) as caught:
                self.post(**options)
            self.assertEqual(caught.exception.code,400)
        with self.assertRaises(HTTPError) as caught:self.raw_post(logprobs=2)
        self.assertEqual(caught.exception.code,400)

    def test_stream_keeps_scores_when_utf8_text_is_buffered(self):
        with self.post(messages=[{'role':'user','content':'unicode'}],logprobs=True,stream=True) as response:
            chunks=[json.loads(value) for value in events(response) if value!='[DONE]']
        records=[];text=''
        for chunk in chunks:
            for choice in chunk['choices']:
                text+=choice['delta'].get('content','')
                if choice.get('logprobs'):records.extend(choice['logprobs']['content'])
        self.assertEqual(text,'🦙')
        self.assertEqual(len(records),4)
        self.assertEqual(bytes(b for record in records for b in record['bytes']).decode(),text)

    def raw_post(self, **changes):
        body = {'model': 'a', 'prompt': 'hello', **changes}
        return urlopen(Request(self.url + '/v1/completions', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'}), timeout=5)

    def test_raw_completion_protocol(self):
        with self.raw_post(prompt=[1, 2]) as response:
            data = json.load(response)
        self.assertEqual(data['object'], 'text_completion')
        self.assertEqual(data['choices'][0]['text'], 'Hello world')
        self.assertNotIn('message', data['choices'][0])
        self.assertEqual(self.runtime.last_options, {'raw_prompt': [1, 2]})
        with self.raw_post(stream=True, stream_options={'include_usage': True}) as response:
            chunks = list(events(response))
        self.assertEqual(chunks[-1], '[DONE]')
        chunks = [json.loads(value) for value in chunks[:-1]]
        self.assertTrue(all(value['object'] == 'text_completion' for value in chunks))
        self.assertEqual(''.join(c['text'] for value in chunks for c in value['choices']), 'Hello world')
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 2)

    def test_raw_completion_rejects_invalid_or_chat_fields(self):
        for change in ({'prompt':None}, {'prompt':[]}, {'prompt':[True]}, {'prompt':['a','b']},
                       {'messages':[]}, {'chat_template_kwargs':{}}, {'max_completion_tokens':3}):
            with self.subTest(change=change), self.assertRaises(HTTPError) as error:
                self.raw_post(**change)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_stream_protocol_usage_and_generator_cleanup(self):
        before = self.runtime.closed
        with self.post(stream=True, stream_options={"include_usage": True}) as response:
            chunks = list(events(response))
        self.assertEqual(chunks[-1], "[DONE]")
        self.assertEqual(json.loads(chunks[-2])["usage"]["completion_tokens"], 2)
        self.assertEqual(self.runtime.closed, before + 1)

    def test_reject_unsupported_parameters(self):
        for changes in ({"temperature": -1}, {"tools": [{'type':'unsupported'}]}, {"stream": "yes"},
                        {"messages": []}, {"stream_options": {"include_usage": 1}}):
            with self.subTest(changes=changes), self.assertRaises(HTTPError) as error:
                self.post(**changes)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_queue_full_is_503_before_stream(self):
        with self.assertRaises(HTTPError) as error:
            self.post(model="busy", stream=True)
        self.assertEqual(error.exception.code, 503)
        error.exception.close()

    def test_unknown_model_is_400(self):
        with self.assertRaises(HTTPError) as error:
            self.post(model="missing", stream=True)
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_models_and_health(self):
        with urlopen(self.url + "/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "a")
        with urlopen(self.url + "/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")

    def test_thinking_option_forwarded_without_coercion(self):
        with self.post(chat_template_kwargs={"enable_thinking": False}) as response:
            self.assertEqual(response.status, 200)
            response.read()
        self.assertEqual(self.runtime.last_options, {"enable_thinking": False})
        for options in ({"enable_thinking": "false"}, {"other": True}, None):
            with self.subTest(options=options), self.assertRaises(HTTPError) as error:
                self.post(chat_template_kwargs=options)
            self.assertEqual(error.exception.code, 400)

    def test_sampling_options_validated_and_forwarded(self):
        options = {'temperature': .7, 'top_p': .9, 'top_k': 20, 'seed': 42}
        with self.post(**options) as response:
            response.read()
        self.assertEqual(self.runtime.last_options, options)
        for change in ({'temperature':True}, {'temperature':float('nan')},
                       {'top_p':0}, {'top_p':1.1}, {'top_k':1.5}, {'seed':-1}):
            with self.subTest(change=change), self.assertRaises(HTTPError) as error:
                self.post(**change)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_stop_and_completion_limit_alias(self):
        with self.post(stop=['END', '🦙'], max_completion_tokens=3) as response:
            response.read()
        self.assertEqual(self.runtime.last_options, {'stop':['END', '🦙']})
        self.assertEqual(self.runtime.last_max_tokens, 3)
        for changes in ({'max_completion_tokens':True}, {'max_tokens':0},
                        {'max_tokens':3, 'max_completion_tokens':3}, {'stop':[]}, {'stop':''}):
            with self.subTest(changes=changes), self.assertRaises(HTTPError) as error:
                self.post(**changes)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_response_format_validation_and_forwarding(self):
        fmt={'type':'json_object'}
        for method in (self.post,self.raw_post):
            with method(response_format=fmt) as response:
                response.read()
            self.assertEqual(self.runtime.last_options['response_format'],fmt)
            with self.assertRaises(HTTPError) as error:
                method(response_format={'type':'json_schema','json_schema':{'name':'bad','schema':{'type':'string','pattern':'x'}}})
            self.assertEqual(error.exception.code,400)
            error.exception.close()

    def test_unsupported_schema_literals_return_400_before_dispatch(self):
        for schema in ({'type':'string','const':'\ud800'},
                       {'type':'integer','const':2**63+1},
                       {'type':'number','enum':[-(2**63)-1]}):
            fmt={'type':'json_schema','json_schema':{'name':'bad','schema':schema}}
            for method in (self.post,self.raw_post):
                for stream in (False,True):
                    with self.subTest(schema=repr(schema),method=method.__name__,stream=stream), \
                            patch.object(self.runtime,'generate') as generate:
                        with self.assertRaises(HTTPError) as caught:
                            method(response_format=fmt,stream=stream)
                        with caught.exception as response:self.assertEqual(response.code,400)
                        generate.assert_not_called()

    def test_management_status_failures_are_structured_and_redacted(self):
        for error,status,message in [(RuntimeError('private backend details'),500,'model management failed'),
                                     (BusyError('private queue details'),503,'model management is busy')]:
            with self.subTest(status=status), patch.object(self.runtime,'manage',side_effect=error):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(self.url+'/v1/models/status',timeout=2)
                with caught.exception as response:
                    self.assertEqual(response.code,status)
                    self.assertEqual(json.load(response),{'error':message})

    def test_management_protocol_and_field_validation(self):
        with urlopen(self.url+'/v1/models/status') as response:
            self.assertEqual(json.load(response)['models'][0]['residency'],'disk')
        for action,body in (('register',{'model':'b','path':'/local/model','chat_template':'{{ messages }}'}),
                            ('load',{'model':'b'}),('unload',{'model':'b','release_host':True})):
            with urlopen(Request(self.url+'/v1/models/'+action,data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})) as response:
                self.assertEqual(response.status,200)
                response.read()
            self.assertEqual(self.runtime.last_management['action'],action)
            if action=='register':self.assertEqual(self.runtime.last_management['chat_template'],'{{ messages }}')
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url+'/v1/models/load',data=b'{"model":"a","path":"wrong"}',
                            headers={'Content-Type':'application/json'}))
        self.assertEqual(error.exception.code,400)
        error.exception.close()

    def test_generation_routes_reject_invalid_unicode_before_dispatch(self):
        for text in ('\ud800', '\udfff'):
            bodies = [('completions', {'prompt':text}),
                      ('completions', {'prompt':'valid', 'stop':text}),
                      ('chat/completions', {'messages':[{'role':'user', 'content':text}]}),
                      ('chat/completions', {'messages':[{'role':'assistant', 'content':'ok', 'reasoning_content':text}]}),
                      ('token-scores', {'prompt':text, 'candidate_token_ids':[1, 2]})]
            for route, fields in bodies:
                with self.subTest(route=route, fields=ascii(fields)), \
                        patch.object(self.runtime, 'generate', side_effect=AssertionError('must not dispatch')) as generate, \
                        patch.object(self.runtime, 'score', side_effect=AssertionError('must not dispatch')) as score:
                    request = Request(self.url+'/v1/'+route,
                        data=json.dumps({'model':'a', **fields}).encode(),
                        headers={'Content-Type':'application/json'})
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(request, timeout=5)
                    self.assertEqual(caught.exception.code, 400)
                    caught.exception.close()
                    generate.assert_not_called()
                    score.assert_not_called()

    def test_encoder_routes_reject_invalid_unicode(self):
        for value in ('\ud800', '\udfff'):
            for route, fields in (('embeddings', {'input':['valid', value]}),
                                  ('rerank', {'query':value, 'documents':['valid']}),
                                  ('rerank', {'query':'valid', 'documents':[value]})):
                with self.subTest(route=route, fields=ascii(fields)):
                    request = Request(self.url+'/v1/'+route,
                        data=json.dumps({'model':'a', **fields}).encode(),
                        headers={'Content-Type':'application/json'})
                    with self.assertRaises(HTTPError) as caught:
                        with urlopen(request, timeout=5):
                            pass
                    self.assertEqual(caught.exception.code, 400)
                    caught.exception.close()

    def test_embedding_protocol_and_validation(self):
        def post(body):
            return urlopen(Request(self.url+'/v1/embeddings',data=json.dumps(body).encode(),
                                  headers={'Content-Type':'application/json'}))
        with post({'model':'a','input':['hello','world'],'encoding_format':'base64','dimensions':384}) as response:
            self.assertEqual(len(json.load(response)['data']),2)
        self.assertEqual(self.runtime.last_embedding,('a',['hello','world'],'base64',384))
        for body in ({'model':'a','input':[True]}, {'model':'a','input':'x','stream':True}, {'input':'x'}):
            with self.assertRaises(HTTPError) as error:post(body)
            self.assertEqual(error.exception.code,400);error.exception.close()

    def test_token_controls_on_both_completion_routes(self):
        options={'presence_penalty':.5,'frequency_penalty':-.5,
                 'repetition_penalty':1.2,'logit_bias':{'1':-100}}
        for post in (self.post,self.raw_post):
            with post(**options) as response:
                self.assertEqual(response.status,200)
            for key,value in options.items():
                self.assertEqual(self.runtime.last_options[key],value)
            for bad in ({'presence_penalty':3},{'logit_bias':{'x':1}},{'frequency_penalty':True}):
                with self.assertRaises(HTTPError) as error:
                    post(**bad)
                self.assertEqual(error.exception.code,400)
                error.exception.close()

    def test_forced_tool_protocol(self):
        tools=[{'type':'function','function':{'name':'weather','strict':True,'parameters':{'type':'object'}}}]
        for choice in ('required',{'type':'function','function':{'name':'weather'}}):
            with self.post(messages=[{'role':'user','content':'forced'}],tools=tools,tool_choice=choice) as response:
                item=json.load(response)['choices'][0]
            self.assertEqual(item['finish_reason'],'tool_calls')
            self.assertEqual(item['message']['tool_calls'][0]['function'],{'name':'weather','arguments':'{}'})
            self.assertEqual(self.runtime.last_options['tool_choice'],choice)
        with self.assertRaises(HTTPError) as caught:
            self.post(tools=tools,tool_choice='required',chat_template_kwargs={'enable_thinking':True})
        self.assertEqual(caught.exception.code,400)

    def test_direct_scoring_protocol_and_validation(self):
        def post(body):
            return urlopen(Request(self.url+'/v1/token-scores',data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}),timeout=5)
        body={'model':'a','prompt':[1,2],'candidate_token_ids':[3,4]}
        with post(body) as response:
            self.assertEqual(json.load(response)['normalization'],'candidate_set')
        self.assertEqual(self.runtime.last_scoring,('a',[1,2],[3,4]))
        for invalid in ({**body,'candidate_token_ids':[3,3]},{**body,'candidate_token_ids':[True,4]},
                {**body,'model':'missing'},{**body,'stream':True},{'model':'a'},[]):
            with self.subTest(body=invalid),self.assertRaises(HTTPError) as caught:post(invalid)
            self.assertEqual(caught.exception.code,400)

    def test_rerank_protocol(self):
        def post(body):
            return urlopen(Request(self.url+'/v1/rerank',data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}),timeout=5)
        body={'model':'a','query':'question','documents':['first','second'],'top_n':1,'return_documents':True}
        with post(body) as response:
            self.assertEqual(json.load(response)['results'],[{'index':1,'relevance_score':1.,'document':{'text':'second'}}])
        for invalid in ({**body,'documents':[]},{**body,'top_n':3},{**body,'return_documents':1},
                {**body,'model':'unknown'},{**body,'stream':True},{'model':'a'},[]):
            with self.subTest(body=invalid),self.assertRaises(HTTPError) as caught:post(invalid)
            self.assertEqual(caught.exception.code,400)

    def test_malformed_reranker_output_is_not_a_client_error(self):
        from dllm.reranking import ranked
        body={'model':'a','query':'question','documents':['first','second']}
        for scores in (1., [[1.], [2.]], [None, 2.]):
            with self.subTest(scores=scores):
                def rerank(*args, **kwargs):
                    return {'results':ranked(scores, body['documents'], 2, False)}
                with patch.object(self.runtime, 'rerank', side_effect=rerank), \
                        patch('dllm.server.traceback.print_exc'):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(Request(self.url+'/v1/rerank', data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'}), timeout=5)
                    with caught.exception as response:
                        self.assertEqual(response.code, 500)
                        self.assertEqual(json.load(response)['error'], 'model execution failed')

    def test_harmony_native_calls_do_not_parse_literal_qwen_envelopes(self):
        tools=[{'type':'function','function':{'name':'weather'}}]
        messages=[{'role':'user','content':'harmony tools'}]
        with self.post(messages=messages,tools=tools) as response:whole=json.load(response)
        choice=whole['choices'][0];message=choice['message']
        self.assertEqual(choice['finish_reason'],'tool_calls')
        self.assertEqual(message['content'],'<tool_call>{"name":"weather","arguments":{}}</tool_call>')
        self.assertEqual(len(message['tool_calls']),1)
        self.assertEqual(message['tool_calls'][0]['function']['arguments'],'{"city":"Paris"}')
        with self.post(messages=messages,tools=tools,stream=True) as response:
            payloads=[json.loads(line[6:]) for line in response if line.startswith(b'data: ') and b'[DONE]' not in line]
        calls=[call for item in payloads for c in item['choices'] for call in c['delta'].get('tool_calls',[])]
        self.assertEqual(calls,[{'index':0,**message['tool_calls'][0]}])
