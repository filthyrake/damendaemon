"""Exact layout and preparation-cost comparison; no inference speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import triton.testing

from dllm.gguf_model import GgufWeights
from scripts.experiments.q4_marlin import Library
from scripts.experiments.q4_marlin_repack import repack


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--library',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4)
    library=Library(args.library)
    source=GgufWeights(args.model,512,'none')
    shifts=torch.tensor([0,4,1,5,2,6,3,7],device='cuda')*4
    def digest(path):
        with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
    report=dict(passed=False,device=torch.cuda.get_device_name(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_sha256=digest(args.model),library_sha256=digest(args.library),
        sources={p:digest(p) for p in ('scripts/experiments/q4_marlin_repack.py',
                                     'scripts/experiments/bench_q4_marlin_repack.py')},
        scope='Bitwise equality with the qualified Marlin repacker. Warm byte-permutation pipeline only: scales, zeros, validation syncs and model activation excluded. No inference speed claim.',cases=[])
    try:
        for name in ('0.qkv.weight','0.out','0.gate_up','0.down','head'):
            weight,scales=source.weights[name].cuda(),source.scales[name].cuda()
            n,k=weight.shape[0],weight.shape[1]*2
            expected=library.prepare(weight,scales).weight
            def original():
                blocks=weight.reshape(n,k//32,16).to(torch.int32)
                values=torch.cat((blocks&15,blocks>>4),dim=-1).reshape(n,k).T.contiguous()
                awq=(values.reshape(k,n//8,8)<<shifts).sum(-1,dtype=torch.int32).contiguous()
                output=torch.empty((k//16,n*2),device=weight.device,dtype=torch.int32)
                library.check(library.lib.dllm_q4_marlin_repack(awq.data_ptr(),output.data_ptr(),
                    k,n,library.sms,library.shared,torch.cuda.current_stream().cuda_stream))
                return output
            functions={'original_pipeline':original}
            functions.update({f'direct_{block}':lambda block=block:repack(weight,block) for block in (256,512,1024)})
            case=dict(name=name,outputs=n,width=k,checks={},peak_additional_bytes={},samples_ms={key:[] for key in functions})
            report['cases'].append(case)
            for key,function in functions.items():
                actual=function()
                exact=bool(torch.equal(actual,expected))
                case['checks'][key]=exact
                assert exact,(name,key,'packed words changed')
                del actual
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated()
                actual=function();torch.cuda.synchronize()
                case['peak_additional_bytes'][key]=torch.cuda.max_memory_allocated()-before
                del actual
            for key in (*functions,*reversed(functions)):
                case['samples_ms'][key].append(triton.testing.do_bench_cudagraph(functions[key],rep=100))
            case['median_ms']={key:statistics.median(values) for key,values in case['samples_ms'].items()}
            print(name,'passed',flush=True)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            del weight,scales,expected,functions
        report['passed']=True
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
