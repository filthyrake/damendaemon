"""Pinned HF prompt rendering and official Rust Harmony stream comparisons."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from openai_harmony import load_harmony_encoding,HarmonyEncodingName,Role,StreamableParser
from dllm.qwen2 import ModelWeights
from dllm.harmony import TextStream


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--output',required=True);parser.add_argument('--generated');args=parser.parse_args()
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    model=ModelWeights(args.model,512)
    reference=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows=[]
    histories=[
        [{'role':'user','content':'What is 2 + 2?'}],
        [{'role':'system','content':'Answer briefly in French.'},{'role':'user','content':'Bonjour, café 🐈.'}],
        [{'role':'user','content':'My name is Ada.'},{'role':'assistant','content':'Hello, Ada.','reasoning_content':'Greet Ada.'},{'role':'user','content':'What is my name?'}],
    ]
    for messages in histories:
        for effort in (None,'low','medium','high'):
            kwargs={} if effort is None else {'reasoning_effort':effort}
            actual=model.chat_tokens(messages,**kwargs)
            expected=reference.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False,**kwargs)
            assert actual==expected,(messages,effort)
            rows.append({'case':'prompt','messages':messages,'reasoning_effort':effort,'tokens':len(actual),'exact':True})
    encoding=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    def text(value):return model.tokenizer.encode(value,add_special_tokens=False).ids
    controls={'start':200006,'message':200008,'end':200007,'return':200002,'channel':200005,'format':200003}
    def frame(channel,body):return [controls['channel']]+text(channel)+[controls['message']]+text(body)
    cases=[
        [controls['message']]+text('A plain answer.')+[controls['return']],
        frame('analysis','Let me think: café, 東京, 🐈.')+[controls['end'],controls['start']]+text('assistant')+
            frame('final','The answer is 42. 🐈')+[controls['return']],
        frame('commentary','Here is the result. ')+[controls['end'],controls['start']]+text('assistant')+
            frame('final','Done.')+[controls['return']],
        [controls['format']]+text('json')+[controls['message']]+text('{"result":"café"}')+[controls['return']],
        frame('analysis','A longer trace with Unicode café 🐈. '*64)+[controls['end'],controls['start']]+text('assistant')+
            frame('final','Finished.')+[controls['return']],
    ]
    generated=json.loads(Path(args.generated).read_text())['rows'] if args.generated else []
    synthetic_count=len(cases);cases.extend(row['tokens'] for row in generated)
    for index,tokens in enumerate(cases):
        native=TextStream(model.tokenizer);official=StreamableParser(encoding,role=Role.ASSISTANT,strict=True)
        ours={'content':'','analysis':''};expected={'content':'','analysis':''}
        for token in tokens:
            content,analysis,_=native.feed(token)
            ours['content']+=content;ours['analysis']+=analysis
            official.process(token)
            delta=official.last_content_delta
            if delta:
                key='analysis' if official.current_channel=='analysis' else 'content'
                expected[key]+=delta
        assert ours==expected,(index,ours,expected)
        if index>=synthetic_count:
            actual=generated[index-synthetic_count]
            assert ours=={'content':actual['content'],'analysis':actual['reasoning_content']}
        assert native.closed
        rows.append({'case':'official_stream','index':index,'tokens':len(tokens),'exact':True,'result':ours})
    result={'passed':True,'model':args.model,'reference_transformers':importlib.metadata.version('transformers'),
        'reference_openai_harmony':importlib.metadata.version('openai-harmony'),'rows':rows,
        'scope':'CPU format checks only; no GPU chat-quality or tool-call qualification follows.'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(rows)}))


if __name__=='__main__':main()
