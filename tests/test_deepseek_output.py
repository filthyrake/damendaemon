import unittest

from damendaemon.deepseek_output import DeepSeekOutput,parse_completion,MARKERS
from damendaemon.deepseek_text import EOS


class DeepSeekOutputTests(unittest.TestCase):
    def test_every_chunk_boundary_matches_literal_control(self):
        for thinking,text,expected in (
            (False,'你好🦙 < ordinary text'+EOS,{'content':'你好🦙 < ordinary text','reasoning_content':''}),
            (True,'Reasoning.\n</think>Answer.'+EOS,{'content':'Answer.','reasoning_content':'Reasoning.\n'}),
            (True,'</think>'+EOS,{'content':'','reasoning_content':''})):
            self.assertEqual(parse_completion(text,enable_thinking=thinking),expected)
            partitions=[[text[:split],text[split:]] for split in range(len(text)+1)]+[list(text)]
            for chunks in partitions:
                parser=DeepSeekOutput(enable_thinking=thinking)
                actual={'content':'','reasoning_content':''}
                for chunk in chunks:
                    for event in parser.feed(chunk):
                        for channel,value in event.items():actual[channel]+=value
                    self.assertLess(len(parser.pending),max(map(len,MARKERS)))
                self.assertEqual(parser.finish(),[])
                self.assertEqual(actual,expected)

    def test_malformed_and_unsupported_controls_poison_parser(self):
        for thinking,text in ((True,'missing terminator'+EOS),(False,'x</think>y'+EOS),
                              (False,'<think>x</think>'+EOS),(False,'x<｜DSML｜tool_calls>'),
                              (False,'x｜DSML｜'),(False,'x'+EOS+'trailing'),(False,'<｜User｜>')):
            parser=DeepSeekOutput(enable_thinking=thinking)
            with self.assertRaises(ValueError):parser.feed(text)
            self.assertTrue(parser.failed)
            with self.assertRaises(RuntimeError):parser.feed('later')
        parser=DeepSeekOutput();parser.feed(EOS)
        with self.assertRaises(ValueError):parser.feed('later')

    def test_truncation_is_explicit_and_preserves_unfinished_channel(self):
        self.assertEqual(parse_completion('still reasoning <',enable_thinking=True,truncated=True),
                         {'content':'','reasoning_content':'still reasoning <'})
        self.assertEqual(parse_completion('reason</think>answer',enable_thinking=True,truncated=True),
                         {'content':'answer','reasoning_content':'reason'})
        parser=DeepSeekOutput();parser.feed('incomplete')
        with self.assertRaises(ValueError):parser.finish()
        with self.assertRaises(RuntimeError):parser.finish(truncated=True)

    def test_bounded_storage_and_invalid_input(self):
        parser=DeepSeekOutput()
        self.assertEqual(parser.feed('x'*100000),[{'content':'x'*100000}])
        self.assertEqual(parser.pending,'')
        with self.assertRaises(ValueError):parser.feed('x'*(1024*1024))
        for value in (None,b'bytes',1):
            parser=DeepSeekOutput()
            with self.assertRaises(ValueError):parser.feed(value)

    def test_tool_calls_are_atomic_across_every_chunk_boundary(self):
        from damendaemon.deepseek_tools import render_tool_calls
        calls=[{'type':'function','function':{'name':'weather','arguments':'{"city": "Montréal"}'}}]
        block=render_tool_calls(calls)
        for thinking,prefix in ((False,'Look up weather.'),(True,'Plan.</think>Look up weather.')):
            text=prefix+'\n\n'+block+EOS
            expected={'content':'Look up weather.','reasoning_content':'Plan.' if thinking else '', 'tool_calls':calls}
            self.assertEqual(parse_completion(text,enable_thinking=thinking,allow_tool_calls=True),expected)
            for chunks in [[text[:split],text[split:]] for split in range(len(text)+1)]+[list(text)]:
                parser=DeepSeekOutput(enable_thinking=thinking,allow_tool_calls=True)
                actual={'content':'','reasoning_content':'','tool_calls':[]}
                for chunk in chunks:
                    for event in parser.feed(chunk):
                        self.assertNotIn('tool_calls',event)
                        for key,value in event.items():actual[key]+=value
                for event in parser.finish():
                    for key,value in event.items():actual[key]+=value
                self.assertEqual(actual,expected)
                self.assertTrue(parser.tool_buffer.closed)

    def test_tool_mode_rejects_partial_malformed_and_reasoning_calls(self):
        from damendaemon.deepseek_tools import render_tool_calls
        block=render_tool_calls([{'type':'function','function':{'name':'f','arguments':'{}'}}])
        for text in ('\n\n'+block,'\n\n'+block+EOS+'trailing','\n\n'+block+EOS+EOS,
                     '\n\n'+block.replace('invoke name="f"','invoke bad="f"')+EOS):
            parser=DeepSeekOutput(allow_tool_calls=True)
            parser.feed(text)
            with self.assertRaises(ValueError):parser.finish(truncated=True)
            self.assertTrue(parser.failed)
            with self.assertRaises(RuntimeError):parser.feed('later')
        with self.assertRaises(ValueError):parse_completion('\n\n'+block+EOS)
        with self.assertRaises(ValueError):parse_completion('\n\n'+block+EOS,enable_thinking=True,allow_tool_calls=True)
        parser=DeepSeekOutput(allow_tool_calls=True)
        parser.feed('\n\n'+block)
        with self.assertRaises(ValueError):parser.feed('x'*(1024*1024))
        self.assertTrue(parser.tool_buffer.closed)

    def test_truncated_tool_opener_is_not_emitted_as_text(self):
        from damendaemon.deepseek_output import TOOL_START
        for size in range(3,len(TOOL_START)):
            with self.assertRaises(ValueError):
                parse_completion(TOOL_START[:size],allow_tool_calls=True,truncated=True)
        self.assertEqual(parse_completion('answer\n',allow_tool_calls=True,truncated=True)['content'],'answer\n')
