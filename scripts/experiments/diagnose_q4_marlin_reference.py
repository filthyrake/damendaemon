"""Reproduce the pinned Mistral QKV subnormal-reference diagnosis.

Run from the qualification checkout with the fixture under baselines/ and the
isolated library under results/. This diagnostic does not change runtime policy.
"""
import json
import torch
from damendaemon.gguf_model import GgufWeights
from damendaemon.quantization import dequantize_q4
from damendaemon.quantized_kernels_q4 import linear
from scripts.experiments.q4_marlin import Library

torch.set_num_threads(4);torch.manual_seed(619)
torch.backends.cuda.matmul.allow_tf32=False
source=GgufWeights('baselines/mistral-spm-q4-comparator.gguf',512,'none')
w,s=source.weights['0.qkv.weight'].cuda(),source.scales['0.qkv.weight'].cuda()
n,k=w.shape[0],w.shape[1]*2
oracle=dequantize_q4(w,s,torch.float16)
bias=torch.randn(n,device='cuda',dtype=torch.float16)
x=torch.randn(2,k,device='cuda',dtype=torch.float16)
lib=Library('results/libdamendaemon-q4-marlin-experiment.so');prepared=lib.prepare(w,s)
rows=[]
for exponent in (-14,-8,0,10):
 value=(x.float()*2.**exponent).half()
 actual=lib.linear(value,prepared);control=linear(value,w,s)
 refs={name:torch.nn.functional.linear(value.to(dtype),oracle.to(dtype)).half()
       for name,dtype in [('half',torch.float16),('single',torch.float32),('double',torch.float64)]}
 comparisons={}
 for name,ref in refs.items():
  diff=actual.float()-ref.float()
  comparisons[name]=dict(rms=(diff.square().mean()/ref.float().square().mean().clamp_min(1e-12)).sqrt().item(),max_abs=diff.abs().max().item(),different=int((actual!=ref).sum()),oracle_rms=ref.float().square().mean().sqrt().item(),subnormal_count=int((ref.abs()<2.**-14).sum()))
 rows.append(dict(exponent=exponent,exact_control=bool(torch.equal(actual,control)),comparisons=comparisons))
print(json.dumps(rows,indent=2))
open('results/q4-marlin-underflow-diagnostic-a2.json','w').write(json.dumps(rows,indent=2)+'\n')
