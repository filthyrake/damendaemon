import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_text import FlashText,render_chat
from damendaemon.glm_flash_generation import FlashReferenceGenerator
from damendaemon.glm_flash_model import FlashModel
from damendaemon.glm_flash_reference_loader import load_reference_model
from damendaemon.glm_text import EOS,END_THINK
from tests import test_glm_generation,test_glm_flash_reference_loader


def text_fixture(path):
    ordinary=test_glm_generation.text_fixture(path)
    return FlashText(path,vocab_size=ordinary.vocab_size)


def snapshot(path,text):
    config,_,_=test_glm_flash_reference_loader.fixture(path,vocab_size=text.vocab_size)
    config['text_config'].update(eos_token_id=sorted(text.eos),pad_token_id=text.tokenizer.token_to_id(EOS),bos_token_id=None)
    (path/'config.json').write_text(json.dumps(config))


class ScriptedFlash(FlashModel):
    __init__=test_glm_generation.ScriptedModel.__init__
    logits=test_glm_generation.ScriptedModel.logits
    prefill=test_glm_generation.ScriptedModel.prefill
    forward=test_glm_generation.ScriptedModel.forward


class FlashGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.text=text_fixture(self.path);self.messages=[dict(role='user',content='Hi')]

    def test_pinned_flash_ordinary_template_contract(self):
        f=json.loads((Path(__file__).parent/'fixtures/glm53_flash_text_contract.json').read_text())
        self.assertEqual(f['tokenizer_vocab_size'],154856)
        for case in f['cases']:self.assertEqual(render_chat(case['messages'],**case['options']),case['prompt'])
        for content in ([dict(type='image',url='unused')],None):
            with self.assertRaises(ValueError):self.text.chat_tokens([dict(role='user',content=content)])
        with self.assertRaises(ValueError):self.text.chat_tokens([dict(role='tool',content='unsupported')])

    def test_family_and_nested_model_contract(self):
        model=ScriptedFlash(self.text,'a')
        with self.assertRaises(ValueError):FlashReferenceGenerator(test_glm_generation.ScriptedModel(self.text,'a'),self.text)
        for config in ({'model_type':'glm_moe_dsa'},{'model_type':'glm5_next','text_config':{'model_type':'glm5_next_text','eos_token_id':[0]}},
                       {'model_type':'glm5_next','text_config':{'model_type':'glm5_next_text','max_position_embeddings':2}}):
            model.config=config
            with self.assertRaises(ValueError):FlashReferenceGenerator(model,self.text,max_context=128)

    def test_padded_sampling_stop_and_abandoned_iterator_release(self):
        model=ScriptedFlash(self.text,'analysis'+END_THINK+'answer'+EOS)
        runner=FlashReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate(self.messages,max_new_tokens=40))
        self.assertEqual(''.join(e.get('content','') for e in events),'answer')
        self.assertEqual(events[-1]['finish_reason'],'stop');self.assertTrue(model.states[-1].closed)
        model=ScriptedFlash(self.text,'abcdef');runner=FlashReferenceGenerator(model,self.text,max_context=128)
        iterator=runner.generate(self.messages,max_new_tokens=5);next(iterator);iterator.close()
        self.assertTrue(model.states[-1].closed)

    def test_primary_failure_and_owned_cleanup_retry(self):
        model=ScriptedFlash(self.text,'abcdef');runner=FlashReferenceGenerator(model,self.text,max_context=128)
        iterator=runner.generate(self.messages,max_new_tokens=4);next(iterator);state=model.states[-1]
        with patch.object(model,'forward',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as error:list(iterator)
            self.assertTrue(error.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(model,'prefill',side_effect=AssertionError('new work')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):list(runner.generate(self.messages,max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed)

    def test_loaded_checkpoint_generation_and_context_admission(self):
        snapshot(self.path,self.text);model=load_reference_model(self.path);model.head.zero_()
        runner=FlashReferenceGenerator(model,self.text,max_context=128)
        prompt=self.text.chat_tokens(self.messages);expected=[]
        for _ in range(2):
            logits,state=model.prefill(torch.tensor([prompt+expected]),capacity=len(prompt)+2)
            expected.append(int(logits[0,:self.text.tokenizer_size].argmax()));state.close()
        events=list(runner.generate(self.messages,max_new_tokens=2))
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events),self.text.decode(expected))
        self.assertEqual(events[-1]['finish_reason'],'length')
        with patch.object(model,'prefill',side_effect=AssertionError('prefill before bounds')):
            with self.assertRaises(ValueError):list(runner.generate(self.messages,max_new_tokens=128))
