from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from tokenizers import Tokenizer, models, AddedToken

from dllm.deepseek_text import DeepSeekText, render_chat, BOS, EOS, USER, ASSISTANT, THINK, END_THINK


class DeepSeekTextTests(unittest.TestCase):
    def test_pinned_literal_prompt_contract(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/deepseek_v4_text_contract.json').read_text())
        for case in fixture['cases']:
            self.assertEqual(render_chat(case['messages'],**case['options']),case['prompt'])

    def test_literal_single_and_multi_turn_modes_and_immutability(self):
        messages=[{'role':'system','content':'Be clear.'},{'role':'user','content':'Hello'},
                  {'role':'assistant','content':'Hi','reasoning_content':'Prior reasoning.'},
                  {'role':'user','content':'Continue'}]
        before=deepcopy(messages)
        base=BOS+'Be clear.'+USER+'Hello'+ASSISTANT
        self.assertEqual(render_chat(messages),base+END_THINK+'Hi'+EOS+USER+'Continue'+ASSISTANT+END_THINK)
        self.assertEqual(render_chat(messages,enable_thinking=True),base+END_THINK+'Hi'+EOS+USER+'Continue'+ASSISTANT+THINK)
        self.assertEqual(render_chat(messages,enable_thinking=True,drop_thinking=False),
                         base+THINK+'Prior reasoning.'+END_THINK+'Hi'+EOS+USER+'Continue'+ASSISTANT+THINK)
        self.assertEqual(messages,before)
        self.assertEqual(render_chat([{'role':'user','content':'你好'}],enable_thinking=True),BOS+USER+'你好'+ASSISTANT+THINK)

    def test_unsupported_modes_and_malformed_histories_fail_explicitly(self):
        valid=[{'role':'user','content':'Hi'}]
        for options in ({'enable_thinking':1},{'drop_thinking':None},{'reasoning_effort':'max'}):
            with self.assertRaises(ValueError):render_chat(valid,**options)
        for messages in ([],[{'role':'system','content':'Only system'}],[{'role':'assistant','content':'Hi'}],
                         valid+valid,[{'role':'developer','content':'Hi'}],
                         [{'role':'user','content':[{'type':'text','text':'Hi'}]}],
                         [{'role':'user','content':'Hi','tools':[]}],
                         [{'role':'user','content':'Hi','reasoning_content':'No'}],
                         [{'role':'user','content':'x'*(1024*1024+1)}]):
            with self.assertRaises(ValueError):render_chat(messages)

    def test_complete_rendered_byte_budget_precedes_tokenization(self):
        cases=[([{'role':'user','content':'你好🙂'}],{}),
               ([{'role':'system','content':'system'},{'role':'user','content':'first'},
                 {'role':'assistant','content':'answer','reasoning_content':'reason'},
                 {'role':'user','content':'next'}],{'enable_thinking':True,'drop_thinking':False})]
        for messages,options in cases:
            before=deepcopy(messages)
            expected=render_chat(messages,**options)
            size=len(expected.encode('utf-8'))
            text=object.__new__(DeepSeekText)
            text.tokenizer=Mock()
            text.tokenizer.encode.return_value.ids=[1,2]
            with self.subTest(options=options):
                with patch('dllm.deepseek_text.MAX_TEXT_BYTES',size-1):
                    with self.assertRaises(ValueError):text.chat_tokens(messages,**options)
                    text.tokenizer.encode.assert_not_called()
                with patch('dllm.deepseek_text.MAX_TEXT_BYTES',size):
                    self.assertEqual(render_chat(messages,**options),expected)
                    self.assertEqual(text.chat_tokens(messages,**options),[1,2])
                    text.tokenizer.encode.assert_called_once_with(expected,add_special_tokens=False)
                self.assertEqual(messages,before)

    def test_local_tokenizer_controls_raw_ids_decode_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            special=(BOS,EOS,USER,ASSISTANT,THINK,END_THINK)
            vocabulary={token:i for i,token in enumerate((*special,'Hi','[UNK]'))}
            tokenizer=Tokenizer(models.WordLevel(vocabulary,unk_token='[UNK]'))
            tokenizer.add_special_tokens([AddedToken(t,special=True) for t in special])
            tokenizer.save(str(path/'tokenizer.json'))
            config=dict(add_bos_token=False,add_eos_token=False,clean_up_tokenization_spaces=False,
                        tokenizer_class='PreTrainedTokenizerFast',bos_token=BOS,eos_token=EOS,pad_token=EOS)
            (path/'tokenizer_config.json').write_text(json.dumps(config))
            text=DeepSeekText(path,vocab_size=8)
            self.assertEqual(text.chat_tokens([{'role':'user','content':'Hi'}]),[0,2,6,3,5])
            self.assertEqual(text.raw_tokens('Hi'),[6])
            self.assertEqual(text.decode([0,6,1]),'Hi')
            self.assertEqual(text.eos,{1})
            ids=[0,6];self.assertIsNot(text.raw_tokens(ids),ids)
            for bad in ([True],[-1],[8],[],None):
                with self.assertRaises(ValueError):text.raw_tokens(bad)
            with self.assertRaises(ValueError):DeepSeekText(path,vocab_size=9)
            config['add_bos_token']=True
            (path/'tokenizer_config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):DeepSeekText(path,vocab_size=8)
