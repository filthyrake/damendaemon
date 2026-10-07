import json
import unittest
from unittest.mock import patch
from dllm.deepseek_tools import parse_tool_calls,render_tool_calls,START,END


def call(name,arguments):
    return {'type':'function','function':{'name':name,'arguments':json.dumps(arguments,ensure_ascii=False)}}


class DeepSeekToolsTests(unittest.TestCase):
    def test_literal_wire_format_and_typed_arguments(self):
        calls=[call('weather',{'city':'Montréal','days':2,'unit':None}),call('clock',{})]
        expected='''<｜DSML｜tool_calls>
<｜DSML｜invoke name="weather">
<｜DSML｜parameter name="city" string="true">Montréal</｜DSML｜parameter>
<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>
<｜DSML｜parameter name="unit" string="false">null</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="clock">

</｜DSML｜invoke>
</｜DSML｜tool_calls>'''
        self.assertEqual(render_tool_calls(calls),expected)
        self.assertEqual(parse_tool_calls(expected),calls)
        values={'s':'<ordinary> & "quoted"\n🙂','false':False,'array':[1,True,{'x':'é'}],'float':1.25}
        self.assertEqual(parse_tool_calls(render_tool_calls([call('nested',values)])),[call('nested',values)])

    def test_malformed_and_every_truncation_rejected(self):
        text=render_tool_calls([call('f',{'x':1})])
        for end in range(len(text)):
            with self.assertRaises(ValueError,msg=str(end)):parse_tool_calls(text[:end])
        for value in (START+END,text+'extra',text.replace('string="false"','string="maybe"'),
                      text.replace('>1</','>NaN</'),text.replace('>1</','>1e999</'),
                      text.replace('>1</','>{"a":1,"a":2}</'),text.replace('>1</','>oops</'),
                      text.replace('</｜DSML｜parameter>\n','</｜DSML｜parameter>'),
                      text.replace('</｜DSML｜invoke>','<｜DSML｜parameter name="x" string="false">2</｜DSML｜parameter>\n</｜DSML｜invoke>')):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_tool_calls(value)

    def test_input_validation_and_resource_bounds(self):
        for arguments in ('[]','null','{"a":1,"a":2}','{"x":NaN}','{"x":1e999}'):
            with self.assertRaises(ValueError):render_tool_calls([{'type':'function','function':{'name':'f','arguments':arguments}}])
        for name in ('','x" bad','x\n', 'x'*129):
            with self.assertRaises(ValueError):render_tool_calls([call(name,{})])
        for value in ('｜DSML｜','</think>','<｜end▁of▁sentence｜>','\ud800'):
            with self.assertRaises(ValueError):render_tool_calls([call('f',{'x':value})])
        with self.assertRaises(ValueError):render_tool_calls([call('f',{})]*65)
        with self.assertRaises(ValueError):render_tool_calls([call('f',{str(i):i for i in range(257)})])
        text=render_tool_calls([call('f',{'x':1})])
        with patch('dllm.deepseek_tools.MAX_TEXT_BYTES',len(text.encode())-1):
            with self.assertRaises(ValueError):parse_tool_calls(text)
            with self.assertRaises(ValueError):render_tool_calls([call('f',{'x':1})])
        self.assertEqual(len(parse_tool_calls(render_tool_calls([call('f',{})]*64))),64)
