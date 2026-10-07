import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer,models,decoders,pre_tokenizers,AddedToken

from dllm.deepseek_generation import BYTE_VALUES,TokenStream,ReferenceGenerator
from dllm.deepseek_text import DeepSeekText,BOS,EOS,USER,ASSISTANT,THINK,END_THINK
from dllm.floating_model import FloatingModel


def text_fixture(path):
    vocabulary={char:i for i,char in enumerate(BYTE_VALUES)}
    tokenizer=Tokenizer(models.BPE(vocab=vocabulary,merges=[]))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(t,special=True) for t in (BOS,EOS,USER,ASSISTANT,THINK,END_THINK)])
    tokenizer.save(str(path/'tokenizer.json'))
    config=dict(add_bos_token=False,add_eos_token=False,clean_up_tokenization_spaces=False,
                tokenizer_class='PreTrainedTokenizerFast',bos_token=BOS,eos_token=EOS,pad_token=EOS)
    (path/'tokenizer_config.json').write_text(json.dumps(config))
    return DeepSeekText(path,vocab_size=262)


class State:
    def __init__(self):self.closed=False
    def close(self):self.closed=True


class ScriptedModel(FloatingModel):
    def __init__(self,text,completion):
        self.embedding=torch.zeros(text.vocab_size,1)
        self.tokens=text.tokenizer.encode(completion,add_special_tokens=False).ids
        self.states=[]
    def logits(self,index):
        result=torch.zeros(1,self.embedding.shape[0]);result[0,self.tokens[index]]=10
        return result
    def prefill(self,tokens,**options):
        self.index=0;state=State();self.states.append(state)
        return self.logits(0),state
    def decode(self,tokens,state,*,position):
        self.index+=1
        return self.logits(self.index)


class ReferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.text=text_fixture(Path(temp.name))
        self.messages=[{'role':'user','content':'Hello'}]

    def test_loaded_packed_model_generation_matches_fresh_prefix_logits(self):
        from tests.test_deepseek_reference_loader import fixture
        from dllm.deepseek_reference_loader import load_reference_model
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path,vocab_size=self.text.vocab_size)
            model=load_reference_model(path)
            # Restrict the fixture head to two ordinary text tokens so this
            # lifecycle control is independent of untrained control-token output.
            model.head['head.weight'].zero_()
            a,b=(self.text.tokenizer.token_to_id(s) for s in ('a','b'))
            model.head['head.weight'][a].fill_(.01)
            model.head['head.weight'][b].fill_(-.01)
            prompt=self.text.chat_tokens(self.messages)
            expected=[]
            for _ in range(3):
                logits,state=model.prefill(torch.tensor([prompt+expected]))
                expected.append(int(logits[0].argmax()));state.close()
            runner=ReferenceGenerator(model,self.text,max_context=128)
            events=list(runner.generate(self.messages,max_new_tokens=3))
            self.assertEqual(''.join(e.get('content','') for e in events),self.text.decode(expected))
            self.assertEqual(events[-1]['finish_reason'],'length')

    def test_fragmented_unicode_and_channels_match_full_tokenizer_decode(self):
        for thinking,completion in ((False,'你好🦙\n'+EOS),(True,'分析🦙</think>回答！'+EOS)):
            stream=TokenStream(self.text,enable_thinking=thinking)
            output={'content':'','reasoning_content':''}
            tokens=self.text.tokenizer.encode(completion,add_special_tokens=False).ids
            with patch.object(self.text,'decode',side_effect=AssertionError('whole-history decode')):
                for token in tokens:
                    for event in stream.feed(token):
                        for key,value in event.items():output[key]+=value
                self.assertEqual(stream.finish(),[])
            self.assertEqual(output,{'content':'回答！' if thinking else '你好🦙\n','reasoning_content':'分析🦙' if thinking else ''})

    def test_eos_length_and_cleanup_before_final_event(self):
        model=ScriptedModel(self.text,'analysis</think>answer'+EOS)
        runner=ReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate(self.messages,max_new_tokens=40,enable_thinking=True))
        self.assertEqual(''.join(e.get('reasoning_content','') for e in events),'analysis')
        self.assertEqual(''.join(e.get('content','') for e in events),'answer')
        self.assertEqual(events[-1]['finish_reason'],'stop')
        self.assertTrue(model.states[-1].closed)
        model=ScriptedModel(self.text,'abcdef')
        runner=ReferenceGenerator(model,self.text,max_context=128)
        events=list(runner.generate(self.messages,max_new_tokens=3))
        self.assertEqual(''.join(e.get('content','') for e in events),'abc')
        self.assertEqual(events[-1]['finish_reason'],'length')
        self.assertEqual(events[-1]['usage']['completion_tokens'],3)
        self.assertTrue(model.states[-1].closed)

    def test_cancellation_and_model_or_parser_failures_release_cache(self):
        model=ScriptedModel(self.text,'abcdef')
        runner=ReferenceGenerator(model,self.text,max_context=128)
        generation=runner.generate(self.messages,max_new_tokens=6)
        next(generation);generation.close()
        self.assertTrue(model.states[-1].closed)
        with patch.object(model,'decode',side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError,'decode failed'):
                list(runner.generate(self.messages,max_new_tokens=6))
        self.assertTrue(model.states[-1].closed)
        model=ScriptedModel(self.text,THINK)
        runner=ReferenceGenerator(model,self.text,max_context=128)
        with self.assertRaises(ValueError):list(runner.generate(self.messages,max_new_tokens=3))
        self.assertTrue(model.states[-1].closed)

    def test_validation_precedes_prefill_and_failed_cleanup_remains_owned(self):
        model=ScriptedModel(self.text,'abcdef')
        runner=ReferenceGenerator(model,self.text,max_context=128)
        with patch.object(model,'prefill',side_effect=AssertionError('prefill started')):
            for options in ({'max_new_tokens':True},{'max_new_tokens':129},{'enable_thinking':1}):
                with self.assertRaises(ValueError):list(runner.generate(self.messages,**options))
        generation=runner.generate(self.messages,max_new_tokens=2)
        next(generation)
        state=model.states[-1]
        original=state.close
        with patch.object(state,'close',side_effect=RuntimeError('release failed')):
            with self.assertRaisesRegex(RuntimeError,'release failed'):list(generation)
            self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(model,'prefill',side_effect=AssertionError('rebuild started')):
                with self.assertRaisesRegex(RuntimeError,'release failed'):
                    list(runner.generate(self.messages,max_new_tokens=2))
        runner.cleanup()
        self.assertTrue(state.closed)
        self.assertFalse(runner.failed_cleanups)

    def test_declared_model_token_ids_and_context_must_match(self):
        model=ScriptedModel(self.text,'a')
        for config in ({'eos_token_id':0},{'eos_token_id':True},{'bos_token_id':0},
                       {'max_position_embeddings':32}):
            model.config=config
            with self.assertRaises(ValueError):ReferenceGenerator(model,self.text,max_context=128)
        model.config={'eos_token_id':list(self.text.eos),'bos_token_id':self.text.tokenizer.token_to_id(BOS),
                      'max_position_embeddings':128}
        ReferenceGenerator(model,self.text,max_context=128)

    def test_tool_token_stream_opt_in_preserves_added_dsml_marker(self):
        from dllm.deepseek_tools import render_tool_calls
        self.text.tokenizer.add_special_tokens([AddedToken('｜DSML｜',special=True)])
        self.text.vocab_size=self.text.tokenizer.get_vocab_size()
        calls=[{'type':'function','function':{'name':'f','arguments':'{"x": "🙂"}'}}]
        completion='Checking.\n\n'+render_tool_calls(calls)+EOS
        tokens=self.text.tokenizer.encode(completion,add_special_tokens=False).ids
        stream=TokenStream(self.text,allow_tool_calls=True)
        events=[]
        for token in tokens:events.extend(stream.feed(token))
        self.assertEqual(''.join(event.get('content','') for event in events),'Checking.')
        self.assertEqual(stream.finish(),[{'tool_calls':calls}])
        stream=TokenStream(self.text)
        with self.assertRaises(ValueError):
            for token in tokens:stream.feed(token)

    def test_reference_tools_finish_reason_cleanup_and_undeclared_rejection(self):
        from dllm.deepseek_tools import render_tool_calls
        tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}]
        calls=[{'type':'function','function':{'name':'f','arguments':'{"x": "🙂"}'}}]
        completion='\n\n'+render_tool_calls(calls)+EOS
        model=ScriptedModel(self.text,completion)
        runner=ReferenceGenerator(model,self.text)
        events=list(runner.generate(self.messages,tools=tools,max_new_tokens=512))
        self.assertIn({'tool_calls':calls},events)
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        self.assertTrue(model.states[0].closed)
        tools[0]['function']['name']='other'
        with self.assertRaises(ValueError):list(runner.generate(self.messages,tools=tools,max_new_tokens=512))
        self.assertTrue(model.states[-1].closed)
        with self.assertRaises(ValueError):list(runner.generate(self.messages,tools=tools,max_new_tokens=30))
        self.assertTrue(model.states[-1].closed)
