"""Compare native packed Q4 kernels against a preserved pre-change module."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import importlib.util
import json
import statistics
import sys
import torch
import triton as tr
import triton.testing
from damendaemon.runtime import require_gpu
from damendaemon.quantization import quantize_q4,dequantize_q4
from damendaemon import quantized_kernels_q4 as current
from damendaemon.quantized_kernels_q4 import linear


def profile_qwen_body(tune=False,adjacent=False):
    from scripts.experiments.q4_prefill import single_dot,single_dot_paired_load,single_dot_adjacent,repack_adjacent
    report=dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),tuning=tune,adjacent=adjacent,checks=[],
                scope='Synthetic real-GPU Qwen2.5-0.5B body matrix profiling; not whole-model performance')
    kernels={name:(kernel,64,128,4,3) for name,kernel in
             (('paired',current._gemm),('contiguous',single_dot),('paired-load',single_dot_paired_load))}
    if tune and not adjacent:
        for bn,warps,stages in ((16,4,2),(32,4,1),(32,4,2),(32,4,3),
                               (64,4,1),(64,8,1),(64,8,2),(64,8,3)):
            kernels[f'load-n{bn}-w{warps}-s{stages}']=(single_dot_paired_load,bn,128,warps,stages)
    if adjacent:
        for bn,warps,stages in ((32,4,1),(64,4,1),(64,4,3)):
            kernels[f'adjacent-n{bn}-w{warps}-s{stages}']=(single_dot_adjacent,bn,128,warps,stages)
        for bn,bk,warps,stages in ((32,64,4,3),(64,64,4,3),(64,64,4,1),(128,64,4,3),
                                  (128,128,4,3),(128,128,8,3),(64,128,2,3),(64,128,8,3)):
            kernels[f'adjacent-n{bn}-k{bk}-w{warps}-s{stages}']=(single_dot_adjacent,bn,bk,warps,stages)
    shapes=((9728,896),(896,4864)) if tune else ((1152,896),(896,896),(9728,896),(896,4864))
    for n,k in shapes:
        weight,scales=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
        reordered=repack_adjacent(weight) if adjacent else None
        restored=dequantize_q4(weight,scales,torch.float16)
        for rows in ((1,4) if tune else (1,2,4,16,64)):
            x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
            expected=torch.nn.functional.linear(x,restored)
            output=torch.empty_like(expected);bm=max(16,tr.next_power_of_2(rows))
            functions={};values={};case=dict(outputs=n,width=k,rows=rows,kernels={})
            for name,(kernel,bn,bk,warps,stages) in kernels.items():
                matrix=reordered if name.startswith('adjacent-') else weight
                def launch(kernel=kernel,bn=bn,bk=bk,warps=warps,stages=stages,matrix=matrix):
                    return kernel[(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,matrix,scales,matrix,output,
                        rows,n,k,bm,bn,bk,False,num_warps=warps,num_stages=stages)
                try:compiled=launch()
                except Exception as error:
                    if not name.startswith(('load-','adjacent-')):raise
                    torch.cuda.synchronize()  # A broken CUDA context must still abort the job.
                    case['kernels'][name]=dict(passed=False,error=repr(error));continue
                values[name]=output.clone()
                error=((output.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                valid=error<.001
                if not name.startswith(('load-','adjacent-')):assert valid,(n,k,rows,name,error)
                case['kernels'][name]=dict(passed=valid,relative_rms=error,registers=compiled.n_regs,spills=compiled.n_spills,round_ms=[])
                if valid:functions[name]=launch
            case['paired_load_matches_contiguous']=bool(torch.equal(values['contiguous'],values['paired-load']))
            assert case['paired_load_matches_contiguous'],(n,k,rows,'contiguous arithmetic changed')
            for name in list(functions):
                if name.startswith(('load-','adjacent-')):
                    exact=bool(torch.equal(values['contiguous'],values[name]))
                    case['kernels'][name]['exact_contiguous_output']=exact
                    if not exact:case['kernels'][name]['passed']=False;del functions[name]
            for name in (*functions,*reversed(functions)):
                case['kernels'][name]['round_ms'].append(tr.testing.do_bench_cudagraph(functions[name],rep=100))
            for value in case['kernels'].values():
                if value.get('round_ms'):value['median_ms']=statistics.median(value['round_ms'])
            report['checks'].append(case)
            print(json.dumps(case),file=sys.stderr,flush=True)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline')
    p.add_argument('--body-candidates',action='store_true')
    p.add_argument('--tune-body',action='store_true')
    p.add_argument('--adjacent-body',action='store_true')
    args=p.parse_args()
    require_gpu();torch.set_num_threads(4);torch.manual_seed(43)
    if args.body_candidates or args.tune_body or args.adjacent_body:
        return profile_qwen_body(args.tune_body or args.adjacent_body,args.adjacent_body)
    if not args.baseline:p.error('--baseline is required unless --body-candidates is selected')
    spec=importlib.util.spec_from_file_location('q4_baseline',args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    reports=[]
    for n,k in ((6144,4096),(28672,4096),(4096,14336),(32000,4096)):
        source=torch.randn(n,k,device='cuda',dtype=torch.float16)*.02
        weight,scales=quantize_q4(source)
        del source
        restored=dequantize_q4(weight,scales,torch.float16)
        for rows in (1,2,4,23):
            x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
            expected=torch.nn.functional.linear(x,restored)
            entry={'outputs':n,'width':k,'rows':rows}
            for name,fn in (('before',baseline.linear),('after',linear)):
                actual=fn(x,weight,scales)
                relative=((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                assert relative<.001,(entry,name,relative)
                entry[name+'_relative_rms']=relative
                entry[name+'_ms']=triton.testing.do_bench_cudagraph(lambda:fn(x,weight,scales),rep=100)
                module=baseline if name=='before' else current
                if rows==1 and name=='before':
                    bk=tr.next_power_of_2(k);bn=2 if bk>=4096 else 8
                    compiled=module._gemv[(tr.cdiv(n,bn),)](x,weight,scales,weight,actual,n,k,rows,bn,bk,False,
                        num_warps=4,enable_fp_fusion=False)
                else:
                    bm=16 if rows<=16 else tr.next_power_of_2(rows);bk=128 if rows<=16 or name=='after' else 64
                    compiled=module._gemm[(tr.cdiv(rows,bm),tr.cdiv(n,64))](x,weight,scales,weight,actual,
                        rows,n,k,bm,64,bk,False,num_warps=8 if bm==128 else 4,num_stages=3)
                entry[name+'_registers']=compiled.n_regs
                entry[name+'_spills']=compiled.n_spills
            print(json.dumps(entry),file=sys.stderr,flush=True)
            reports.append(entry)
        del restored,weight,scales
    print(json.dumps({'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'scope':'warm CUDA-graph matrix kernels; not whole-model serving','checks':reports},indent=2))


if __name__=='__main__':main()
