import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer,models,decoders,pre_tokenizers,AddedToken
from dllm.bytelevel_stream import BYTE_VALUES
from dllm.glm_text import GlmText,CONTROLS,EOS,USER,OBSERVATION,THINK,END_THINK
from dllm.glm_output import GlmOutput
from dllm.glm_generation import TokenStream,ReferenceGenerator
from dllm.glm_model import GlmModel
from tests.test_deepseek_generation import State


def text_fixture(path):
    vocabulary={char:i for i,char in enumerate(BYTE_VALUES)}
    tokenizer=Tokenizer(models.BPE(vocab=vocabulary,merges=[]))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(t,special=True) for t in (*CONTROLS,'<tool_call>')])
    tokenizer.save(str(path/'tokenizer.json'))
    config=dict(backend='tokenizers',tokenizer_class='TokenizersBackend',clean_up_tokenization_spaces=False,
                do_lower_case=False,remove_space=False,eos_token=EOS,pad_token=EOS)
    (path/'tokenizer_config.json').write_text(json.dumps(config))
    generation=dict(eos_token_id=[tokenizer.token_to_id(t) for t in (EOS,USER,OBSERVATION)],pad_token_id=tokenizer.token_to_id(EOS))
    (path/'generation_config.json').write_text(json.dumps(generation))
    return GlmText(path,vocab_size=tokenizer.get_vocab_size()+24)


class ScriptedModel(GlmModel):
    def __init__(self,text,completion):
        self.embedding=torch.zeros(text.vocab_size,1)
        self.tokens=text.tokenizer.encode(completion,add_special_tokens=False).ids
        self.states=[];self.failed_cleanups=[];self.lock=threading.RLock()
    def logits(self,index):
        result=torch.zeros(1,self.embedding.shape[0]);result[0,self.tokens[index]]=10
        result[0,-1]=100 # padded row must never be sampled
        return result
    def prefill(self,tokens,**options):
        self.index=0;state=State();self.states.append(state);return self.logits(0),state
    def forward(self,tokens,state,*,position):
        self.index+=1;return self.logits(self.index)


class GlmGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.text=text_fixture(Path(temp.name));self.messages=[dict(role='user',content='Hi')]

    def test_every_text_split_and_byte_fragment_matches_channels(self):
        for stop in (EOS,USER,OBSERVATION):
            completion='分析🦙'+END_THINK+'回答！'+stop
            for split in range(len(completion)+1):
                parser=GlmOutput();events=parser.feed(completion[:split])+parser.feed(completion[split:])+parser.finish()
                self.assertEqual(''.join(e.get('reasoning_content','') for e in events),'分析🦙')
                self.assertEqual(''.join(e.get('content','') for e in events),'回答！')
            stream=TokenStream(self.text);events=[]
            with patch.object(self.text,'decode',side_effect=AssertionError('whole decode')):
                for token in self.text.raw_tokens(completion):events.extend(stream.feed(token))
                events.extend(stream.finish())
            self.assertEqual(''.join(e.get('content','') for e in events),'回答！')

    def test_padded_rows_stop_length_and_cleanup_before_completion(self):
        model=ScriptedModel(self.text,'analysis'+END_THINK+'answer'+EOS)
        runner=ReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate(self.messages,max_new_tokens=40))
        self.assertEqual(''.join(e.get('content','') for e in events),'answer')
        self.assertEqual(events[-1]['finish_reason'],'stop');self.assertTrue(model.states[-1].closed)
        model=ScriptedModel(self.text,'abcdef');runner=ReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate(self.messages,max_new_tokens=3))
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events),'abc')
        self.assertEqual(events[-1]['finish_reason'],'length');self.assertTrue(model.states[-1].closed)

    def test_cancel_decode_failure_and_unsupported_controls_release_cache(self):
        model=ScriptedModel(self.text,'abcdef');runner=ReferenceGenerator(model,self.text,max_context=128)
        generation=runner.generate(self.messages,max_new_tokens=6);next(generation);generation.close()
        self.assertTrue(model.states[-1].closed)
        with patch.object(model,'forward',side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError,'decode failed'):list(runner.generate(self.messages,max_new_tokens=6))
        self.assertTrue(model.states[-1].closed)
        for completion in (THINK,EOS,END_THINK+'<tool_call>'):
            model=ScriptedModel(self.text,completion);runner=ReferenceGenerator(model,self.text,max_context=128)
            with self.assertRaises(ValueError):list(runner.generate(self.messages,max_new_tokens=6))
            self.assertTrue(model.states[-1].closed)
        stream=TokenStream(self.text)
        with self.assertRaises(ValueError):stream.feed(self.text.tokenizer_size)

    def test_failed_cleanup_remains_owned_and_validation_precedes_prefill(self):
        model=ScriptedModel(self.text,'abcdef');runner=ReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'prefill',side_effect=AssertionError('prefill')):
            for options in ({'max_new_tokens':True},{'max_new_tokens':128},{'reasoning_effort':'medium'}):
                with self.assertRaises(ValueError):list(runner.generate(self.messages,**options))
        generation=runner.generate(self.messages,max_new_tokens=2);next(generation);state=model.states[-1]
        with patch.object(state,'close',side_effect=RuntimeError('release failed')):
            with self.assertRaisesRegex(RuntimeError,'release failed'):list(generation)
            self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(model,'prefill',side_effect=AssertionError('new prefill')):
                with self.assertRaisesRegex(RuntimeError,'release failed'):list(runner.generate(self.messages,max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed);self.assertFalse(runner.failed_cleanups)

    def test_model_token_contract_and_nonfinite_logits(self):
        model=ScriptedModel(self.text,'a')
        for config in ({'eos_token_id':[0]},{'bos_token_id':0},{'pad_token_id':True},{'max_position_embeddings':2}):
            model.config=config
            with self.assertRaises(ValueError):ReferenceGenerator(model,self.text,max_context=128)
        model.config={};runner=ReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'logits',return_value=torch.full((1,self.text.vocab_size),float('nan'))):
            with self.assertRaisesRegex(ValueError,'nonfinite'):list(runner.generate(self.messages,max_new_tokens=1))
        self.assertTrue(model.states[-1].closed)

    def test_loaded_packed_snapshot_generation_matches_fresh_prefix(self):
        from tests.test_glm_reference_loader import fixture
        from dllm.glm_reference_loader import load_reference_model
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path,vocab_size=self.text.vocab_size);model=load_reference_model(path)
            model.head.zero_() # ordinary byte zero wins; no untrained control-token claims
            prompt=self.text.chat_tokens(self.messages);expected=[]
            for _ in range(3):
                logits,state=model.prefill(torch.tensor([prompt+expected]),capacity=len(prompt)+3)
                expected.append(int(logits[0,:self.text.tokenizer_size].argmax()));state.close()
            runner=ReferenceGenerator(model,self.text,max_context=128)
            events=list(runner.generate(self.messages,max_new_tokens=3))
            self.assertEqual(''.join(e.get('reasoning_content','') for e in events),self.text.decode(expected))
            self.assertEqual(events[-1]['finish_reason'],'length')

    def test_parser_failure_lifecycle_and_byte_bound(self):
        for value in (END_THINK+END_THINK,END_THINK+EOS+'trailing','<tool_call>'):
            parser=GlmOutput()
            with self.assertRaises(ValueError):parser.feed(value)
            with self.assertRaises(RuntimeError):parser.finish(truncated=True)
        parser=GlmOutput();parser.feed('reason')
        with self.assertRaises(ValueError):parser.finish()
        with patch('dllm.glm_output.MAX_TEXT_BYTES',4):
            parser=GlmOutput()
            with self.assertRaisesRegex(ValueError,'byte limit'):parser.feed('hello')

    def test_primary_decode_error_survives_failed_cleanup(self):
        model=ScriptedModel(self.text,'abcdef');runner=ReferenceGenerator(model,self.text,max_context=128)
        generation=runner.generate(self.messages,max_new_tokens=3);next(generation);state=model.states[-1]
        with patch.object(model,'forward',side_effect=RuntimeError('primary decode failure')), \
                patch.object(state,'close',side_effect=RuntimeError('secondary close failure')):
            with self.assertRaisesRegex(RuntimeError,'primary decode failure') as error:list(generation)
        self.assertTrue(error.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
        runner.cleanup();self.assertTrue(state.closed)
