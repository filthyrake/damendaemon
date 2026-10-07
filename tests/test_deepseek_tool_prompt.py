import copy
import json
import unittest
from pathlib import Path

from dllm.deepseek_text import render_chat,BOS,USER,ASSISTANT,THINK,END_THINK,EOS
from dllm.deepseek_tool_prompt import tool_schema_prompt
from dllm.deepseek_tools import render_tool_calls

TOOLS=[{'type':'function','function':{'name':'weather','description':'Get weather','parameters':{'type':'object'}}}]


class DeepSeekToolPromptTests(unittest.TestCase):
    def test_pinned_literal_prompt_fixture(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/deepseek_v4_tool_prompt.json').read_text())
        self.assertEqual(render_chat(fixture['messages'],tools=fixture['tools']),fixture['prompt'])

    def test_ordered_results_reasoning_retention_and_no_input_mutation(self):
        calls=[{'id':name,'type':'function','function':{'name':'weather','arguments':json.dumps({'city':name})}} for name in ('A','B')]
        messages=[{'role':'user','content':'Weather?'},
                  {'role':'assistant','content':None,'reasoning_content':'Check both.','tool_calls':calls},
                  {'role':'tool','tool_call_id':'B','content':'Rain'},
                  {'role':'tool','tool_call_id':'A','content':'Sun'},
                  {'role':'user','content':'Compare.'}]
        original=copy.deepcopy(messages)
        schema,_=tool_schema_prompt(TOOLS)
        expected=(BOS+'\n\n'+schema+USER+'Weather?'+ASSISTANT+THINK+'Check both.'+END_THINK+
                  '\n\n'+render_tool_calls(calls)+EOS+USER+
                  '<tool_result>Sun</tool_result>\n\n<tool_result>Rain</tool_result>\n\nCompare.'+ASSISTANT+THINK)
        self.assertEqual(render_chat(messages,tools=TOOLS,enable_thinking=True),expected)
        self.assertEqual(messages,original)
        chat=render_chat(messages[:-1],tools=TOOLS)
        self.assertTrue(chat.endswith('<tool_result>Rain</tool_result>'+ASSISTANT+END_THINK))
        self.assertNotIn('Check both.',chat)

    def test_schema_and_history_rejections(self):
        for tools in ([],TOOLS*2,[{'type':'other','function':{}}],[{'type':'function','function':{'name':'f','parameters':[]}}],
                      [{'type':'function','function':{'name':'f','strict':1}}]):
            with self.assertRaises(ValueError):render_chat([{'role':'user','content':'hi'}],tools=tools)
        call={'id':'a','type':'function','function':{'name':'weather','arguments':'{}'}}
        base=[{'role':'user','content':'hi'},{'role':'assistant','tool_calls':[call]}]
        for results in ([],[{'role':'tool','tool_call_id':'wrong','content':'ok'}],
                        [{'role':'tool','tool_call_id':'a','content':'ok'}]*2,
                        [{'role':'tool','tool_call_id':'a','content':'</tool_result>'}]):
            with self.assertRaises(ValueError):render_chat(base+results,tools=TOOLS)
        for change in ({'id':None},{'id':'a','function':{'name':'unknown','arguments':'{}'}}):
            bad=copy.deepcopy(base);bad[1]['tool_calls'][0].update(change)
            with self.assertRaises(ValueError):render_chat(bad+[{'role':'tool','tool_call_id':'a','content':'ok'}],tools=TOOLS)
