from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from tokenizers import Tokenizer,AddedToken
from dllm.glm_flash_image_text import FlashImageText,render_image_chat,BEGIN_IMAGE,IMAGE,END_IMAGE
from tests.test_glm_text import tokenizer_fixture


def adapter(path):
    tokenizer_fixture(path);tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
    tokenizer.add_special_tokens([AddedToken(s,special=True) for s in (BEGIN_IMAGE,IMAGE,END_IMAGE)])
    tokenizer.save(str(path/'tokenizer.json'))
    return FlashImageText(path,vocab_size=tokenizer.get_vocab_size()+24)


class FlashImageTextTests(unittest.TestCase):
    def test_pinned_official_image_prompts_and_immutability(self):
        data=json.loads(Path('tests/fixtures/glm53_flash_image_text_contract.json').read_text())
        for case in data['cases']:
            messages=deepcopy(case['messages'])
            self.assertEqual(render_image_chat(messages,**case['options']),case['prompt'])
            self.assertEqual(messages,case['messages'])
            self.assertEqual(case['token_ids'].count(154854),sum(g[1]*g[2]//4 for g in case['grids']))

    def test_expansion_order_and_ordinary_text_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            text=adapter(Path(temp));messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text='Hi'),dict(type='image')])]
            actual=text.image_chat_tokens(messages,[[1,2,4],[1,4,4]])
            prompt=render_image_chat(messages);parts=prompt.split(IMAGE)
            expanded=parts[0]+IMAGE*2+parts[1]+IMAGE*4+parts[2]
            self.assertEqual(actual,text.tokenizer.encode(expanded,add_special_tokens=False).ids)
            self.assertEqual(actual.count(text.image_token_id),6)
            with self.assertRaises(ValueError):text.chat_tokens(messages)
            ordinary=[dict(role='user',content='Hi')]
            self.assertEqual(text.image_chat_tokens(ordinary,[]),text.chat_tokens(ordinary))

    def test_reject_unsupported_parts_spoofing_and_invalid_grids(self):
        bad_contents=([dict(type='image_url',image_url='https://example.test/a.png')],[dict(type='video')],
                      [dict(type='audio')],[dict(type='image',data='ignored')],[dict(type='text',text=IMAGE)],IMAGE,[])
        for content in bad_contents:
            with self.assertRaises(ValueError):render_image_chat([dict(role='user',content=content)])
        with tempfile.TemporaryDirectory() as temp:
            text=adapter(Path(temp));messages=[dict(role='user',content=[dict(type='image')])]
            for grids in ([],[[2,2,2]],[[True,2,2]],[[1,3,2]],[[1,2,2],[1,2,2]]):
                with self.assertRaises(ValueError):text.image_chat_tokens(messages,grids)
            text.tokenizer=Mock(wraps=text.tokenizer)
            with self.assertRaisesRegex(ValueError,'budget'):text.image_chat_tokens(messages,[[1,65536,65536]])
            text.tokenizer.encode.assert_not_called()
            with patch('dllm.glm_flash_image_text.MAX_TOKENS',4):
                with self.assertRaisesRegex(ValueError,'token limit'):text.image_chat_tokens(messages,[[1,2,2]])

    def test_atomic_markers_and_text_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);tokenizer_fixture(path)
            with self.assertRaisesRegex(ValueError,'atomic'):FlashImageText(path,vocab_size=64)
        with self.assertRaises(ValueError):render_image_chat([dict(role='user',content=[dict(type='image')]*257)])
        with self.assertRaises(ValueError):render_image_chat([dict(role='user',content=[dict(type='text',text='x'*(1024**2+1))])])
