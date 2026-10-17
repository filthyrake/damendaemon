"""Guarded tiny DeepSeek CUDA routing/packed FFN against decoded CPU control."""
import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from torch.nn import functional as F
    from damendaemon.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
    from damendaemon.cuda_deepseek_ffn import CudaPackedDeepSeekFeedForward
    from damendaemon.cuda_packed_moe import cuda_packed_moe
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid call admitted')
    try:
        rng=torch.Generator().manual_seed(249)
        hidden=torch.randn(2,3,32,generator=rng)*.2
        router=torch.randn(3,32,generator=rng)*.3
        bias=torch.tensor([.2,-.1,.3]);table=torch.tensor([[0,1],[1,2],[2,0],[0,2]])
        ids=torch.tensor([[0,1,2],[3,1,0]])
        for format in ('e4m3fn','e2m1'):
            def matrix(rows,columns):
                if format=='e4m3fn':
                    raw=(torch.randn(rows,columns,generator=rng)*.2).to(torch.float8_e4m3fn).view(torch.uint8)
                    scales=torch.full(((rows+127)//128,(columns+127)//128),127,dtype=torch.uint8)
                    decoded=raw.view(torch.float8_e4m3fn).float()
                else:
                    codes=torch.randint(0,16,(rows,columns),generator=rng,dtype=torch.uint8)
                    raw=codes[:,::2]|(codes[:,1::2]<<4)
                    scales=torch.full((rows,(columns+31)//32),124,dtype=torch.uint8)
                    lookup=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
                    decoded=lookup[codes.long()]*.125
                return CudaPackedMatrix(raw.cuda(),scales.cuda(),format),decoded
            bank=[];control=[]
            for _ in range(4):
                matrices=[matrix(32,32),matrix(32,32),matrix(32,32)]
                bank.append(CudaPackedExpert(*(m[0] for m in matrices),row_chunk=16,column_chunk=16,workspace_bytes=65536,activation_limit=.1))
                control.append([m[1] for m in matrices])
            for hashed in (False,True):
                options=dict(router=router.cuda(),shared=bank[-1],top_k=2,activation_limit=.1,route_scale=1.5)
                options.update(token_table=table.cuda()) if hashed else options.update(correction_bias=bias.cuda())
                ffn=CudaPackedDeepSeekFeedForward(bank[:3],**options)
                captured=[]
                def capture(source,experts,selected,coefficients,**kwargs):
                    captured.append((selected.cpu(),coefficients.cpu()))
                    return cuda_packed_moe(source,experts,selected,coefficients,**kwargs)
                with patch('damendaemon.cuda_deepseek_ffn.cuda_packed_moe',side_effect=capture):
                    actual=ffn(hidden.cuda(),ids.cuda() if hashed else None).cpu()
                expected=torch.zeros_like(hidden)
                def expert(x,weights,coefficient=1.):
                    gate,up,down=weights
                    activation=F.silu((x@gate.T).clamp(max=.1))*(x@up.T).clamp(-.1,.1)
                    return (activation*coefficient)@down.T
                for b in range(2):
                    for t in range(3):
                        x=hidden[b,t];prob=F.softplus(router@x).sqrt()
                        selected=table[ids[b,t]] if hashed else torch.argsort(prob+bias,descending=True,stable=True)[:2]
                        coefficients=prob[selected]/prob[selected].sum()*1.5
                        seen_ids,seen_weights=captured[b*3+t]
                        torch.testing.assert_close(seen_ids[0],selected,rtol=0,atol=0)
                        torch.testing.assert_close(seen_weights[0],coefficients,rtol=3e-5,atol=1e-7)
                        expected[b,t]=expert(x,control[-1])
                        for index,weight in zip(selected.tolist(),coefficients):expected[b,t]+=expert(x,control[index],weight)
                torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
                before=hidden.clone();bad=hidden.cuda();bad[0,0,0]=float('nan')
                rejected(lambda:ffn(bad,ids.cuda() if hashed else None))
                if hashed:
                    rejected(lambda:ffn(hidden.cuda(),torch.full_like(ids,-1,device='cuda')))
                    rejected(lambda:ffn(hidden.cuda(),torch.full_like(ids,4,device='cuda')))
                    torch.testing.assert_close(ffn.token_table.cpu(),table,rtol=0,atol=0)
                ffn.max_output_bytes=1
                with patch('damendaemon.cuda_deepseek_ffn.F.linear',side_effect=AssertionError('projection before admission')):
                    rejected(lambda:ffn(hidden.cuda(),ids.cuda() if hashed else None))
                torch.testing.assert_close(hidden,before,rtol=0,atol=0)
                report['cases'].append(dict(format=format,hashed=hashed,ids_exact=True,max_absolute=float((actual-expected).abs().max())))
        report.update(passed=True,bounds_and_invalid_inputs=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
