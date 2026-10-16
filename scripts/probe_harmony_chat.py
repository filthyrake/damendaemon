"""Real native GPT-OSS text chat and token-framing capture for reference parsing."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import torch
from dllm.scheduler import BatchedRuntime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--output',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    runtime=BatchedRuntime({'oss':args.model},context=512,batch_size=2,host_gib=18)
    rows=[]
    try:
        cases=[([{'role':'user','content':'What is 2 + 2? Answer briefly.'}],'low'),
            ([{'role':'user','content':'My name is Ada.'},{'role':'assistant','content':'Hello, Ada.'},
              {'role':'user','content':'What is my name? Answer briefly.'}],'low')]
        for messages,effort in cases:
            events=list(runtime.generate('oss',messages,128,reasoning_effort=effort))
            rows.append({'messages':messages,'reasoning_effort':effort,
                'tokens':[e['token'] for e in events if 'token' in e],
                'content':''.join(e.get('text','') for e in events),
                'reasoning_content':''.join(e.get('reasoning_content','') for e in events),'final':events[-1]})
            print(json.dumps(rows[-1]),flush=True)
        assert '4' in rows[0]['content'] and 'Ada' in rows[1]['content']
        assert all(row['final']['finish_reason']=='stop' and row['reasoning_content'] for row in rows)
        Path(args.output).write_text(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2)+'\n')
    finally:runtime.close()


if __name__=='__main__':main()
