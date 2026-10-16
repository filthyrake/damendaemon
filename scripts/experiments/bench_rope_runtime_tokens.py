"""Real CUDA rotary/cache equivalence and warm-kernel comparison; no model claim."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
import triton as tr
import triton.testing
from dllm.kernels import _rope_store, _rope_store_runtime_tokens
from scripts.experiments.rope_runtime_tokens import rope_store_runtime_tokens


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--integrated',action='store_true');args=p.parse_args()
    assert torch.cuda.device_count()==1,'select exactly one GPU'
    torch.set_num_threads(4);torch.manual_seed(43)
    report=dict(passed=False,integrated=args.integrated,device=torch.cuda.get_device_name(),checks=[],
        scope='Real GPU rotary/query/KV-cache equivalence and warm kernel timing. Qualified stage geometry; no model or HTTP performance claim.',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('dllm/kernels.py','scripts/experiments/rope_runtime_tokens.py',__file__)})
    heads,kv_heads,dim,context=14,2,64,512
    try:
        for norm,precise in ((False,False),(False,True),(True,False),(True,True)):
            for tokens in (1,2,3,5,7,16,17,31,32,33,63,64,65,127,128,129,255):
                x=torch.randn(tokens,(heads+2*kv_heads)*dim,device='cuda',dtype=torch.float16)
                cos=torch.randn(context,dim,device='cuda',dtype=torch.float16);sin=torch.randn_like(cos)
                offset=(context-tokens)//2;positions=torch.arange(offset,offset+tokens,device='cuda')
                qweight=torch.randn(dim,device='cuda',dtype=torch.float16);kweight=torch.randn_like(qweight)
                initial_k=torch.randn(1,kv_heads,context,dim,device='cuda',dtype=torch.float16);initial_v=torch.randn_like(initial_k)
                buffers={name:(torch.empty(1,heads,tokens,dim,device='cuda',dtype=torch.float16),initial_k.clone(),initial_v.clone())
                         for name in ('static','dynamic')}
                functions={};case=dict(tokens=tokens,norm=norm,precise=precise,kernels={})
                dynamic_kernel=_rope_store_runtime_tokens if args.integrated else rope_store_runtime_tokens
                for name,kernel in (('static',_rope_store),('dynamic',dynamic_kernel)):
                    def launch(name=name,kernel=kernel):
                        q,k,v=buffers[name]
                        return kernel[(tokens,heads+2*kv_heads)](x,cos,sin,positions,q,k,v,qweight,kweight,
                            norm,1e-6,precise,tokens,heads,kv_heads,dim,context,dim,enable_fp_fusion=False)
                    compiled=launch();functions[name]=launch
                    case['kernels'][name]=dict(assembly_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),warm_ms=[])
                case['exact_query_kv']=all(torch.equal(a,b) for a,b in zip(buffers['static'],buffers['dynamic']))
                assert case['exact_query_kv'],case
                for name in ('static','dynamic','dynamic','static'):
                    case['kernels'][name]['warm_ms'].append(tr.testing.do_bench_cudagraph(functions[name],rep=50))
                for value in case['kernels'].values():value['median_warm_ms']=statistics.median(value['warm_ms'])
                report['checks'].append(case)
            print('norm',norm,'precise',precise,'passed',flush=True)
        report['passed']=True
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
