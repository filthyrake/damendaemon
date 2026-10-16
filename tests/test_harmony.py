import unittest
from dllm.harmony import TextStream,_MARKERS


class Tokenizer:
    def __init__(self):
        self.special={name:value for value,names in _MARKERS.values() for name in names}
        self.parts={1:b'analysis',2:b'final',3:b'assistant',4:b'let me think',5:b'42',
            6:b' to=functions.bad',7:b'commentary',8:b'json',9:b'caf\xc3',10:b'\xa9',
            11:b'AB',12:b'C rest',13:b'system',14:b' to=all',15:b'analysis <|channel|>'}
    def token_to_id(self,name):return self.special.get(name)
    def decode(self,tokens,skip_special_tokens=False):return b''.join(self.parts[t] for t in tokens).decode(errors='replace')


START=200006;MESSAGE=200008;END=200007;RETURN=200002;CHANNEL=200005;CALL=200012


class HarmonyTests(unittest.TestCase):
    def parse(self,tokens,stops=()):
        parser=TextStream(Tokenizer(),stops);content='';reasoning='';rows=[]
        for i,token in enumerate(tokens):
            text,analysis,stop=parser.feed(token,final=i==len(tokens)-1)
            content+=text;reasoning+=analysis;rows.append((text,analysis,stop))
            if stop:break
        return content,reasoning,rows

    def test_analysis_final_and_control_tokens_are_separated(self):
        content,reasoning,rows=self.parse([CHANNEL,1,MESSAGE,4,END,START,3,CHANNEL,2,MESSAGE,5,RETURN])
        self.assertEqual((content,reasoning),('42','let me think'))
        self.assertTrue(rows[-1][2])
        self.assertFalse(any(r[2] for r in rows[:-1]))

    def test_utf8_partial_tokens_and_length_truncation(self):
        content,_,rows=self.parse([CHANNEL,2,MESSAGE,9,10,RETURN])
        self.assertEqual(content,'café');self.assertEqual(rows[3][0],'')
        self.assertEqual(self.parse([CHANNEL,2,MESSAGE,9])[0],'caf\ufffd')
        self.assertEqual(self.parse([CHANNEL,1])[0:2],('',''))

    def test_stops_ignore_analysis_and_span_message_boundaries(self):
        tokens=[CHANNEL,1,MESSAGE,11,12,END,START,3,CHANNEL,2,MESSAGE,11,END,START,3,CHANNEL,2,MESSAGE,12,RETURN]
        content,reasoning,rows=self.parse(tokens,('ABC',))
        self.assertEqual(content,'');self.assertEqual(reasoning,'ABC rest');self.assertTrue(rows[-1][2])

    def test_plain_and_commentary_messages(self):
        self.assertEqual(self.parse([MESSAGE,5,RETURN])[0],'42')
        self.assertEqual(self.parse([14,CHANNEL,7,MESSAGE,5,RETURN])[0],'42')

    def test_literal_header_words_in_body_are_not_control_tokens(self):
        self.assertEqual(self.parse([MESSAGE,15,RETURN])[0],'analysis <|channel|>')

    def test_assistant_reasoning_history_round_trips_without_mutation(self):
        from dllm.tool_calls import validate_messages,template_messages
        messages=[{'role':'assistant','content':'4','reasoning_content':'Simple.'}]
        self.assertEqual(validate_messages(messages),messages)
        self.assertEqual(template_messages(messages),messages)
        for invalid in ([],{},True):
            with self.assertRaises(ValueError):
                validate_messages([{**messages[0],'reasoning_content':invalid}])
        with self.assertRaises(ValueError):
            validate_messages([{'role':'user','content':'hello','reasoning_content':'x'}])

    def test_bad_roles_recipients_frames_and_controls_fail(self):
        for tokens in ([6,MESSAGE],[MESSAGE,5,CALL],[MESSAGE,START],[CHANNEL,1,CHANNEL],
            [MESSAGE,5,END,5],[200099],[RETURN],[MESSAGE,5,END,START,13,MESSAGE]):
            with self.subTest(tokens=tokens),self.assertRaises(ValueError):self.parse(tokens)
        parser=TextStream(Tokenizer());parser.feed(MESSAGE);parser.feed(RETURN)
        with self.assertRaises(ValueError):parser.feed(5)

    def test_header_is_bounded_and_checkpoint_markers_are_validated(self):
        with self.assertRaises(ValueError):self.parse([3]*257)
        tokenizer=Tokenizer();tokenizer.special.pop('<|start|>')
        with self.assertRaises(ValueError):TextStream(tokenizer)

    def test_declared_handoff_is_buffered_and_complete(self):
        import json
        tokenizer=Tokenizer();tokenizer.parts.update({16:b' to=functions.weather',17:b'{"city":"caf\xc3',18:b'\xa9", "literal":"<tool_call> }"}'})
        tools=[{'type':'function','function':{'name':'weather'}}]
        tokenizer.parts[19]=b' json'
        for header in ([16,CHANNEL,7],[CHANNEL,7,16],[16,CHANNEL,7,19]):
            parser=TextStream(tokenizer,tools=tools)
            for token in header+[MESSAGE,17,18]:
                self.assertEqual(parser.feed(token),('','',False));self.assertEqual(parser.calls,[])
            self.assertEqual(parser.feed(CALL),('','',True))
            self.assertEqual(parser.calls[0]['function']['name'],'weather')
            self.assertEqual(json.loads(parser.calls[0]['function']['arguments']),{'city':'café','literal':'<tool_call> }'})

    def test_invalid_tool_json_and_framing_never_publish_calls(self):
        tokenizer=Tokenizer();tokenizer.parts[16]=b' to=functions.weather'
        tools=[{'type':'function','function':{'name':'weather'}}]
        for body in (b'{',b'[]',b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            tokenizer.parts[17]=body;parser=TextStream(tokenizer,tools=tools)
            for token in [16,CHANNEL,7,MESSAGE,17]:parser.feed(token)
            with self.assertRaises(ValueError):parser.feed(CALL)
            self.assertEqual(parser.calls,[])
        tokenizer.parts[17]=b'{}'
        for end in (END,RETURN,199999):
            parser=TextStream(tokenizer,tools=tools)
            for token in [16,CHANNEL,7,MESSAGE,17]:parser.feed(token)
            with self.assertRaises(ValueError):parser.feed(end)
        parser=TextStream(tokenizer,tools=tools)
        for token in [16,CHANNEL,7,MESSAGE]:parser.feed(token)
        self.assertEqual(parser.feed(17,final=True),('','',False));self.assertEqual(parser.calls,[])
        for header in ([16,CHANNEL,1],[16,16,CHANNEL,7],[6,CHANNEL,7]):
            parser=TextStream(tokenizer,tools=tools)
            with self.assertRaises(ValueError):
                for token in header+[MESSAGE]:parser.feed(token)

    def test_tool_history_normalization_and_unsupported_shapes(self):
        from types import SimpleNamespace
        from dllm.qwen2 import ModelWeights
        from dllm.tool_calls import validate_messages
        class Template:
            def generate(self,**kwargs):self.arguments=kwargs;yield 'rendered'
        template=Template();model=SimpleNamespace(template=template,config={'model_type':'gpt_oss','vocab_size':2},
            supports_tool_calls=True,special_tokens={},tokenizer=SimpleNamespace(encode=lambda *a,**k:SimpleNamespace(ids=[1])))
        call={'id':'call_a','type':'function','function':{'name':'weather','arguments':'{"city":"Paris"}'}}
        history=[{'role':'assistant','content':None,'reasoning_content':'Need weather.','tool_calls':[call]},
            {'role':'tool','tool_call_id':'call_a','content':'Sunny'}]
        tools=[{'type':'function','function':{'name':'weather'}}]
        self.assertEqual(ModelWeights.chat_tokens(model,history,tools=tools),[1])
        normalized=template.arguments['messages'][0]
        self.assertEqual(normalized['content'],'');self.assertEqual(normalized['thinking'],'Need weather.')
        self.assertEqual(normalized['tool_calls'][0]['function']['arguments'],{'city':'Paris'})
        self.assertIsNone(history[0]['content']);self.assertNotIn('thinking',history[0])
        self.assertNotIn('description',tools[0]['function'])
        with self.assertRaisesRegex(ValueError,'auto or none'):ModelWeights.chat_tokens(model,history,tools=tools,tool_json=True)
        with self.assertRaises(ValueError):validate_messages([{**history[0],'reasoning_content':[]},history[1]])
        second={**call,'id':'call_b'}
        multiple=[{**history[0],'tool_calls':[call,second]},history[1],{'role':'tool','tool_call_id':'call_b','content':'Sunny'}]
        with self.assertRaisesRegex(ValueError,'one call'):ModelWeights.chat_tokens(model,multiple,tools=tools)
