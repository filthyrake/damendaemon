from http.server import ThreadingHTTPServer
import json
import threading
import unittest
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

    def generate(self, alias, messages, max_tokens, **kwargs):
        self.last_options = kwargs
        self.last_max_tokens = max_tokens
        try:
            if alias == "busy":
                raise BusyError("request queue is full")
            if alias != "a":
                raise ValueError("unknown model")
            yield {"text": "Hello", "token": 1}
            yield {"text": " world", "token": 2}
            yield {"finish_reason": "length", "usage": {
                "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, "timing": {}}
        finally:
            self.closed += 1


class ServerTests(unittest.TestCase):
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

    def test_nonstream_completion(self):
        with self.post() as response:
            data = json.load(response)
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(data["usage"]["completion_tokens"], 2)

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
        for changes in ({"temperature": -1}, {"tools": []}, {"stream": "yes"},
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

    def test_management_protocol_and_field_validation(self):
        with urlopen(self.url+'/v1/models/status') as response:
            self.assertEqual(json.load(response)['models'][0]['residency'],'disk')
        for action,body in (('register',{'model':'b','path':'/local/model'}),
                            ('load',{'model':'b'}),('unload',{'model':'b','release_host':True})):
            with urlopen(Request(self.url+'/v1/models/'+action,data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})) as response:
                self.assertEqual(response.status,200)
                response.read()
            self.assertEqual(self.runtime.last_management['action'],action)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url+'/v1/models/load',data=b'{"model":"a","path":"wrong"}',
                            headers={'Content-Type':'application/json'}))
        self.assertEqual(error.exception.code,400)
        error.exception.close()

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
