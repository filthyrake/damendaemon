from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from tokenizers import Tokenizer,models,AddedToken
from dllm.glm_text import GlmText,render_chat,CONTROLS,EOS,USER,OBSERVATION,ASSISTANT,THINK


def tokenizer_fixture(path):
    vocabulary={token:index for index,token in enumerate((*CONTROLS,'Hi','[UNK]'))}
    tokenizer=Tokenizer(models.WordLevel(vocabulary,unk_token='[UNK]'))
    tokenizer.add_special_tokens([AddedToken(t,special=True) for t in CONTROLS])
    tokenizer.save(str(path/'tokenizer.json'))
    config=dict(backend='tokenizers',tokenizer_class='TokenizersBackend',clean_up_tokenization_spaces=False,
                do_lower_case=False,remove_space=False,eos_token=EOS,pad_token=EOS)
    (path/'tokenizer_config.json').write_text(json.dumps(config))
    generation=dict(eos_token_id=[vocabulary[t] for t in (EOS,USER,OBSERVATION)],pad_token_id=vocabulary[EOS])
    (path/'generation_config.json').write_text(json.dumps(generation))
    return vocabulary,config,generation


class GlmTextTests(unittest.TestCase):
    def test_pinned_official_literal_prompt_contract_and_immutability(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/glm53_text_contract.json').read_text())
        for case in fixture['cases']:
            messages=deepcopy(case['messages'])
            self.assertEqual(render_chat(messages,**case['options']),case['prompt'])
            self.assertEqual(messages,case['messages'])
            self.assertTrue(case['prompt'].endswith(ASSISTANT+THINK))

    def test_bad_histories_unsupported_features_and_bounds(self):
        valid=[dict(role='user',content='Hi')]
        invalid=([],[dict(role='system',content='only system')],[dict(role='assistant',content='Hi')],valid+valid,
                 [dict(role='user',content=[dict(type='text',text='Hi')])],
                 [dict(role='developer',content='Hi')],[dict(role='tool',content='result')],
                 [dict(role='user',content='Hi',reasoning_content='No')],
                 valid+[dict(role='assistant',content='Hi',tool_calls=[])]+valid,
                 [dict(role='user',content='x'*(1024**2+1))])
        for messages in invalid:
            with self.assertRaises(ValueError):render_chat(messages)
        for options in ({'reasoning_effort':'medium'},{'clear_thinking':1}):
            with self.assertRaises(ValueError):render_chat(valid,**options)
        with self.assertRaises(TypeError):render_chat(valid,enable_thinking=False)

    def test_local_tokenizer_padded_rows_raw_ids_and_special_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);vocabulary,config,generation=tokenizer_fixture(path)
            text=GlmText(path,vocab_size=len(vocabulary)+24)
            self.assertEqual(text.tokenizer_size,len(vocabulary));self.assertEqual(text.eos,set(generation['eos_token_id']))
            self.assertEqual(text.raw_tokens('Hi'),[vocabulary['Hi']])
            self.assertEqual(text.decode([vocabulary['Hi']]),'Hi')
            tokens=text.chat_tokens([dict(role='user',content='Hi')])
            self.assertEqual(tokens[-2:],[vocabulary[ASSISTANT],vocabulary[THINK]])
            ids=[vocabulary['Hi']];self.assertIsNot(ids,text.raw_tokens(ids))
            for bad in ([True],[-1],[len(vocabulary)],[],None):
                with self.assertRaises(ValueError):text.raw_tokens(bad)
            with self.assertRaises(ValueError):text.decode([len(vocabulary)])
            with self.assertRaises(ValueError):GlmText(path,vocab_size=len(vocabulary)-1)

    def test_tokenizer_and_generation_metadata_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);vocabulary,config,generation=tokenizer_fixture(path)
            for change in ({'eos_token_id':[0]},{'pad_token_id':True},{'eos_token_id':generation['eos_token_id']*2}):
                (path/'generation_config.json').write_text(json.dumps(generation|change))
                with self.assertRaises(ValueError):GlmText(path,vocab_size=len(vocabulary))
            (path/'generation_config.json').write_text(json.dumps(generation))
            (path/'tokenizer_config.json').write_text(json.dumps(config|{'do_lower_case':True}))
            with self.assertRaises(ValueError):GlmText(path,vocab_size=len(vocabulary))
            (path/'tokenizer_config.json').write_text(' '*(1024**2+1))
            with self.assertRaisesRegex(ValueError,'byte limit'):GlmText(path,vocab_size=len(vocabulary))
