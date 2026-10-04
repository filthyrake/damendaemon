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
