"""CPU-only tool prompt and handoff comparison against pinned HF and Rust Harmony."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from openai_harmony import load_harmony_encoding,HarmonyEncodingName,Role,StreamableParser
from damendaemon.qwen2 import ModelWeights
from damendaemon.harmony import TextStream
from scripts.verify_harmony_tools import TOOLS,MESSAGES


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--captured',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4)
    model=ModelWeights(args.model,1024);reference=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    captured=json.loads(Path(args.captured).read_text());first=captured['rows'][0]['response']['choices'][0]['message']
    history=MESSAGES+[first,{'role':'tool','tool_call_id':first['tool_calls'][0]['id'],'content':'Paris: 19 degrees Celsius, sunny.'}]
    histories=[MESSAGES,history,history+[{'role':'assistant','content':'It is 19 degrees.'},{'role':'user','content':'What about Tokyo?'}]]
    rows=[]
    for messages in histories:
        for effort in ('low','medium','high'):
            actual=model.chat_tokens(messages,tools=TOOLS,reasoning_effort=effort)
            # Explicit API-to-template adaptation, independently spelled out here.
            normalized=copy.deepcopy(messages)
            for m in normalized:
                if m.get('tool_calls'):
                    m['content']=m.get('content') or ''
                    if m.get('reasoning_content'):m['thinking']=m['reasoning_content']
                    for call in m['tool_calls']:call['function']['arguments']=json.loads(call['function']['arguments'])
            expected=reference.apply_chat_template(normalized,tools=TOOLS,reasoning_effort=effort,
                tokenize=True,add_generation_prompt=True,return_dict=False)
            assert actual==expected
            rows.append({'case':'tool_prompt','messages':len(messages),'effort':effort,'tokens':len(actual),'exact':True})
    encoding=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    text=lambda value:model.tokenizer.encode(value,add_special_tokens=False).ids
    tokens=captured['rows'][-1]['tokens']
    examples=[tokens,
        text(' to=functions.weather')+[200005]+text('commentary json')+[200008]+text('{"city":"café 🐈", "literal":"<tool_call> }"}')+[200012],
        [200005]+text('commentary to=functions.weather')+[200008]+text('{"city":"Tokyo"}')+[200012]]
    for index,tokens in enumerate(examples):
        native=TextStream(model.tokenizer,tools=TOOLS);official=StreamableParser(encoding,role=Role.ASSISTANT,strict=True)
        content='';analysis='';ours_content='';ours_analysis='';arguments='';recipient=None
        for token in tokens:
            c,a,_=native.feed(token);ours_content+=c;ours_analysis+=a
            official.process(token)
            if official.last_content_delta:
                if official.current_recipient and official.current_recipient!='all':
                    arguments+=official.last_content_delta;recipient=official.current_recipient
                elif official.current_channel=='analysis':analysis+=official.last_content_delta
                else:content+=official.last_content_delta
        assert native.closed and len(native.calls)==1
        function=native.calls[0]['function']
        assert recipient=='functions.'+function['name']
        assert json.loads(arguments)==json.loads(function['arguments'])
        assert (ours_content,ours_analysis)==(content,analysis)
        rows.append({'case':'official_handoff','tokens':len(tokens),'exact':True,'function':function})
    result={'passed':True,'checks':len(rows),'transformers':importlib.metadata.version('transformers'),
        'openai_harmony':importlib.metadata.version('openai-harmony'),'rows':rows,'scope':'CPU format comparisons; not a performance benchmark.'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(rows)}))


if __name__=='__main__':main()
