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
