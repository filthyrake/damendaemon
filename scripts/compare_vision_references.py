"""Compare independently generated vision targets across execution devices."""
import argparse
import hashlib
import json
import torch
from pathlib import Path
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument('first')
parser.add_argument('second')
parser.add_argument('--output',required=True)
args=parser.parse_args()
a=torch.load(args.first,weights_only=True)
b=torch.load(args.second,weights_only=True)
if [c['name'] for c in a['cases']] != [c['name'] for c in b['cases']]:
    raise ValueError('reference case sets differ')
report={'scope':'independent Transformers device arithmetic control','cases':[],
        'first_sha256':hashlib.sha256(Path(args.first).read_bytes()).hexdigest(),
        'second_sha256':hashlib.sha256(Path(args.second).read_bytes()).hexdigest()}
for first,second in zip(a['cases'],b['cases']):
    checks=[]
    for name,x in first['targets'].items():
        y=second['targets'][name]
        d=x-y
        checks.append(dict(stage=name,maximum=d.abs().max().item(),relative_rms=(d.square().mean()/x.square().mean()).sqrt().item(),reference_maximum=x.abs().max().item()))
    report['cases'].append(dict(name=first['name'],checks=checks))
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
for case in report['cases']:
 print(case['name'],[c for c in case['checks'] if c['maximum']>=max(.02,5e-4*c['reference_maximum'])])
