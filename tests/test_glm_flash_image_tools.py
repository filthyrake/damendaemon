import io
import json
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from damendaemon.glm_flash_image_tool_text import FlashImageToolText,render_image_tool_chat
from damendaemon.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from damendaemon.glm_flash_inline_images import decode_inline_images
from damendaemon.glm_flash_tools import FlashTools
from damendaemon.glm_text import EOS,END_THINK
from scripts import run_glm_flash_image_reference as cli
from tests.test_glm_flash_tool_generation import TOOLS,CALLS
from tests.test_glm_flash_image_generation import text_fixture,vision,ScriptedImageModel
from tests.test_glm_flash_inline_images import messages as inline_messages,snapshot

FIXTURE=json.loads((Path(__file__).parent/'fixtures/glm53_flash_image_tool_contract.json').read_text())


class FlashImageToolsTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        base=text_fixture(self.path);self.text=FlashImageToolText(self.path,vocab_size=base.vocab_size)
        self.messages=[dict(role='user',content=[dict(type='text',text='Hi'),dict(type='image')])]
        self.images=[torch.full((112,112,3),127,dtype=torch.uint8)]
        self.wire=FlashTools(TOOLS).render(CALLS)

    def runner(self,completion=None):
        model=ScriptedImageModel(self.text,completion or 'reason'+END_THINK+self.wire+EOS)
        return FlashImageToolReferenceGenerator(model,self.text,vision(1),max_context=4096)

    def test_pinned_combined_prompts_and_image_expansion(self):
        for case in FIXTURE['cases']:
            self.assertEqual(render_image_tool_chat(case['messages'],FIXTURE['tools'],**case['options']),case['prompt'])
            tokens=self.text.image_tool_chat_tokens(case['messages'],case['grids'],FIXTURE['tools'],**case['options'])
            self.assertEqual(tokens.count(self.text.image_token_id),16)
            unexpanded=self.text.tokenizer.encode(case['prompt'],add_special_tokens=False).ids
            expected=[]
            for token in unexpanded:expected.extend([token]*16 if token==self.text.image_token_id else [token])
            self.assertEqual(tokens,expected)
        with self.assertRaises(ValueError):self.text.tool_chat_tokens(self.messages,TOOLS)
        with self.assertRaises(ValueError):self.text.chat_tokens(self.messages)

    def test_image_to_tool_generation_suppresses_media_and_resumes_results(self):
        runner=self.runner();events=list(runner.generate_image_tools(self.messages,self.images,TOOLS,max_new_tokens=512))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls');self.assertTrue(runner.model.states[-1].closed)
        calls=next(e['tool_calls'] for e in events if 'tool_calls' in e)
        self.assertEqual(json.loads(calls[0]['function']['arguments']),{'value':'false'})
        prompt=self.text.image_tool_chat_tokens(self.messages,[[1,8,8]],TOOLS)
        self.assertEqual(events[-1]['usage']['prompt_tokens'],len(prompt))
        self.assertEqual(runner.model.last_values.shape,(1,len(prompt),1))
        history=self.messages+[dict(role='assistant',content='',reasoning_content='reason',tool_calls=calls),
                               dict(role='tool',tool_call_id=calls[0]['id'],content='result')]
        following=self.runner('reason'+END_THINK+'done'+EOS)
        events=list(following.generate_image_tools(history,self.images,TOOLS,max_new_tokens=128,clear_thinking=True))
        self.assertEqual(''.join(e.get('content','') for e in events),'done')

    def test_invalid_schema_counts_context_parts_and_text_before_vision(self):
        runner=self.runner()
        with patch.object(runner.vision,'forward',side_effect=AssertionError('vision')),patch.object(runner.model,'prefill_embeddings',side_effect=AssertionError('cache')):
            for messages,images,tools,limit in ((self.messages,[],TOOLS,2),(self.messages,self.images,[],2),
                    (self.messages,self.images,TOOLS,4096),
                    ([dict(role='user',content=[dict(type='text',text='<|image|>')])],self.images,TOOLS,2),
                    ([dict(role='user',content=[dict(type='video')])],self.images,TOOLS,2),
                    ([dict(role='user',content=[dict(type='text',text='x'*600000)]*2)],self.images,TOOLS,2)):
                with self.assertRaises(ValueError):list(runner.generate_image_tools(messages,images,tools,max_new_tokens=limit))
        with self.assertRaises(ValueError):self.text.image_tool_chat_tokens(self.messages,[],TOOLS)

    def test_cancel_and_decode_cleanup_failure_preserve_ownership(self):
        runner=self.runner();stream=runner.generate_image_tools(self.messages,self.images,TOOLS,max_new_tokens=512)
        next(stream);stream.close();self.assertTrue(runner.model.states[-1].closed)
        stream=runner.generate_image_tools(self.messages,self.images,TOOLS,max_new_tokens=512);next(stream);state=runner.model.states[-1]
        with patch.object(runner.model,'forward',side_effect=RuntimeError('decode')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'decode') as error:list(stream)
            self.assertTrue(error.exception.__notes__)
            self.assertEqual(runner.failed_cleanups,[state])
        runner.cleanup();self.assertTrue(state.closed)

    def test_inline_history_validates_before_pixels_and_preserves_flash_names(self):
        source=inline_messages();tools=deepcopy(TOOLS);tools[0]['function']['name']='lookup.long'
        call=deepcopy(CALLS[0]);call['id']='call.0';call['function']['name']='lookup.long'
        source.extend([dict(role='assistant',content=None,tool_calls=[call]),dict(role='tool',content='result',tool_call_id='call.0')])
        original=deepcopy(source);decoded=decode_inline_images(source,tools=tools)
        self.assertEqual(len(decoded.images),1);self.assertEqual(source,original)
        with patch('PIL.Image.open',side_effect=AssertionError('pixels')):
            with self.assertRaises(ValueError):decode_inline_images(source,tools=[])
            with self.assertRaises(ValueError):decode_inline_images(source[:-1],tools=tools)

    def test_actual_combined_snapshot_cli_tool_mode_and_preflight(self):
        snapshot(self.path);messages=self.path/'messages.json';messages.write_text(json.dumps(inline_messages()))
        tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS));output=io.StringIO()
        argv=[str(self.path),'--messages',str(messages),'--tools',str(tools),'--max-new-tokens','2']
        with redirect_stdout(output):self.assertEqual(cli.main(argv),0)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertFalse(events[0]['official_model_qualified']);self.assertEqual(events[0]['images'],1)
        self.assertEqual(events[-1]['finish_reason'],'length')
        tools.write_text('[]')
        with patch.object(cli,'load_reference_model',side_effect=AssertionError('payload')), \
                patch('PIL.Image.open',side_effect=AssertionError('pixels')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(argv)
            self.assertEqual(error.exception.code,2)
