"""Tiny CUDA DeepSeek residual reduction/head gate against scalar equations."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_hyper_connections import reduce_streams,restore_streams,reduce_head
    from tests.test_hyper_connections import HyperConnectionTests,scalar_weights
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        host=HyperConnectionTests().fixture();hidden,projection,scale,bias=[v.cuda() for v in host]
        originals=[v.clone() for v in (hidden,projection,scale,bias)]
        for iterations in (1,20):
            reduced,post,matrix=reduce_streams(hidden,projection,scale,bias,iterations=iterations)
            head=reduce_head(hidden,projection[:4],scale[:1],bias[:4])
            expected=torch.empty_like(host[0][:,:,0]);expected_head=torch.empty_like(expected)
            expected_post=torch.empty(2,3,4);expected_matrix=torch.empty(2,3,4,4)
            for b in range(2):
                for t in range(3):
                    rows=host[0][b,t].tolist();flat=[x for row in rows for x in row]
                    norm=math.sqrt(sum(x*x for x in flat)/len(flat)+1e-6)
                    mixes=[sum(x*w for x,w in zip(flat,row))/norm for row in host[1].tolist()]
                    pre,po,mat=scalar_weights(mixes,host[2].tolist(),host[3].tolist(),4,iterations,1e-6)
                    expected[b,t]=torch.tensor([sum(pre[i]*rows[i][j] for i in range(4)) for j in range(5)])
                    weights=[1/(1+math.exp(-(mixes[i]*float(host[2][0])+float(host[3][i]))))+1e-6 for i in range(4)]
                    expected_head[b,t]=torch.tensor([sum(weights[i]*rows[i][j] for i in range(4)) for j in range(5)])
                    expected_post[b,t]=torch.tensor(po);expected_matrix[b,t]=torch.tensor(mat)
            for a,e in ((reduced,expected),(post,expected_post),(matrix,expected_matrix),(head,expected_head)):
                torch.testing.assert_close(a.cpu(),e,atol=1e-6,rtol=2e-5)
            restored=restore_streams(reduced,hidden,post,matrix)
            control=torch.stack([post.cpu()[:,:,dest,None]*reduced.cpu()+sum(matrix.cpu()[:,:,src,dest,None]*host[0][:,:,src] for src in range(4)) for dest in range(4)],dim=2)
            torch.testing.assert_close(restored.cpu(),control,atol=1e-6,rtol=2e-5)
            for option in ('max_workspace_bytes','max_output_bytes'):
                with patch('dllm.hyper_connections.finite',side_effect=AssertionError('scanned before budget')):
                    rejected(lambda:reduce_streams(hidden,projection,scale,bias,**{option:1}))
                    rejected(lambda:reduce_head(hidden,projection[:4],scale[:1],bias[:4],**{option:1}))
                    rejected(lambda:restore_streams(reduced,hidden,post,matrix,**{option:1}))
            for bad in (projection.cpu(),projection.double(),projection+float('nan')):
                rejected(lambda:reduce_streams(hidden,bad,scale,bias))
            rejected(lambda:reduce_streams(hidden,projection,scale,bias,iterations=0))
            rejected(lambda:reduce_head(hidden,projection[:4],scale[:1],bias[:4],norm_epsilon=float('nan')))
            rejected(lambda:restore_streams(reduced.cpu(),hidden,post,matrix))
            report['cases'].append(dict(iterations=iterations,max_absolute=float((reduced.cpu()-expected).abs().max()),head_max_absolute=float((head.cpu()-expected_head).abs().max()),restore=True))
        for a,b in zip((hidden,projection,scale,bias),originals):torch.testing.assert_close(a,b,atol=0,rtol=0)
        report.update(passed=True,preallocation_bounds=True,inputs_unchanged=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
