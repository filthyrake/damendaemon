import json
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch

from dllm.reference_serving import ReferenceRuntime,handler_for_reference
from dllm.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from dllm.glm_flash_image_tool_text import FlashImageToolText
from dllm.glm_flash_tools import FlashTools
from dllm.glm_text import END_THINK,EOS
from dllm.http_auth import AuthPolicy
from dllm.server import BoundedHTTPServer
from dllm.runtime import BusyError
from tests.test_glm_flash_inline_images import messages as image_messages
from tests.test_glm_flash_image_generation import text_fixture,vision,ScriptedImageModel
from tests.test_reference_serving import TOOLS,CALLS


class ReferenceImageServingTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        text=text_fixture(self.path);text=FlashImageToolText(self.path,vocab_size=text.vocab_size)
        model=ScriptedImageModel(text,'reason'+END_THINK+'answer'+FlashTools(TOOLS).render(CALLS)+EOS)
        self.runner=FlashImageToolReferenceGenerator(model,text,vision(1),max_context=4096)
        self.runtime=ReferenceRuntime(self.runner);self.addCleanup(self.runtime.close)
        self.server=BoundedHTTPServer(('127.0.0.1',0),handler_for_reference(self.runtime,AuthPolicy()))
        thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
        self.addCleanup(lambda:(self.server.shutdown(),self.server.server_close(),thread.join(3)))
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        self.messages=image_messages()

    def post(self,**changes):
        body=dict(model='model',messages=self.messages,tools=TOOLS,max_tokens=512)|changes
        return urlopen(Request(self.url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=5)

    def test_image_tool_http_sse_and_returned_history(self):
        original=deepcopy(self.messages)
        with self.post() as response:result=json.load(response)
        message=result['choices'][0]['message'];call=message['tool_calls'][0]
        self.assertEqual(result['choices'][0]['finish_reason'],'tool_calls')
        self.assertEqual(message['content'],'answer');self.assertEqual(message['reasoning_content'],'reason')
        self.assertTrue(self.runner.model.states[-1].closed);self.assertEqual(self.messages,original)
        self.assertEqual(self.runtime.manage('status')['modality'],'still_image')
        history=self.messages+[message,dict(role='tool',tool_call_id=call['id'],content='result')]
        with self.post(messages=history,stream=True,stream_options={'include_usage':True}) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire);self.assertIn('tool_calls',wire);self.assertIn('cpu_reference',wire)
        self.assertNotIn(call['id'],wire);self.assertTrue(self.runner.model.states[-1].closed)

    def test_rejects_invalid_remote_overbudget_and_history_before_execution(self):
        bad=image_messages();bad[0]['content'][1]['image_url']['url']='https://example.test/image.png'
        malformed=image_messages();malformed[0]['content'][1]['image_url']['url']='data:image/png;base64,AAAA'
        cases=[dict(messages=bad),dict(messages=malformed),dict(messages=image_messages()*5),
               dict(messages=[dict(role='user',content='text only')]),dict(max_tokens=4096),
               dict(messages=self.messages+[dict(role='tool',tool_call_id='missing',content='bad')]),dict(temperature=1)]
        with patch.object(self.runner.vision,'forward',side_effect=AssertionError('vision')),patch.object(self.runner.model,'prefill_embeddings',side_effect=AssertionError('text')):
            for change in cases:
                with self.subTest(change=list(change)),self.assertRaises(HTTPError) as error:self.post(**change)
                self.assertEqual(error.exception.code,400)
            self.runner.vision.max_tokens=4
            with self.assertRaises(HTTPError) as error:self.post()
            self.assertEqual(error.exception.code,400)
        self.assertFalse(self.runner.model.states)

    def test_busy_before_pixels_and_abandonment_releases_image_state(self):
        stream=self.runtime.generate('model',self.messages,512,tools=TOOLS);next(stream)
        with patch('PIL.Image.open',side_effect=AssertionError('pixels before owner')):
            with self.assertRaises(BusyError):next(self.runtime.generate('model',self.messages,512,tools=TOOLS))
        stream.close();self.assertTrue(self.runner.model.states[-1].closed)
        with patch.object(self.runner.vision,'forward',side_effect=RuntimeError('vision failure')):
            with self.assertRaisesRegex(RuntimeError,'vision failure'):list(self.runtime.generate('model',self.messages,512,tools=TOOLS))
        self.assertFalse(self.runtime.active);self.assertFalse(self.runtime.cleanup_pending())
        with self.post() as response:self.assertEqual(response.status,200)
