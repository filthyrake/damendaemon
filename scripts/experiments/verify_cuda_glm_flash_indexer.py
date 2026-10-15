"""Tiny CUDA learned-pool selection gate with scalar score and padding controls."""
import argparse
import math
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_glm_flash_indexer import CudaFlashIndexer
    from dllm.glm_flash_indexer import FlashIndexer
    from tests.test_glm_flash_indexer import indexer
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        c=indexer();weights={n:v.cuda() for n,v in c.weights.items()}
        gpu=CudaFlashIndexer(weights,heads=2,pool_size=2,top_k=2)
        hidden=torch.randn(1,7,4);low=torch.randn(1,7,2);pos=torch.arange(7).reshape(1,7)
        raw=hidden@c.weights['wk.weight'].T
        expected_keys=(raw-raw.mean(-1,keepdim=True))/torch.sqrt(raw.var(-1,unbiased=False,keepdim=True)+1e-6)
        expected_keys=expected_keys*c.weights['k_norm.weight']+c.weights['k_norm.bias']
        expected_gates=hidden@c.weights['index_kpool_compress_gate'].T
        keys,gates=gpu.project_keys(hidden.cuda())
        torch.testing.assert_close(keys.cpu(),expected_keys,atol=1e-6,rtol=3e-6)
        torch.testing.assert_close(gates.cpu(),expected_gates,atol=1e-6,rtol=3e-6)
        selected=gpu.select(hidden.cuda(),low.cuda(),keys,gates,pos.cuda()).cpu()
        for token in range(7):
            scores=[]
            for start in range(0,token,2):
                pooled=[]
                for d in range(3):
                    logits=[float(expected_gates[0,start+j,d]+c.weights['index_kpool_compress_ape'][j,d]) for j in range(2)]
                    probs=[math.exp(v-max(logits)) for v in logits]
                    pooled.append(sum(probs[j]*float(expected_keys[0,start+j,d]) for j in range(2))/sum(probs))
                score=0
                for head in range(2):
                    q=[sum(float(low[0,token,r])*float(c.weights['wq_b.weight'][head*3+d,r]) for r in range(2)) for d in range(3)]
                    weight=sum(float(hidden[0,token,r])*float(c.weights['weights_proj.weight'][head,r]) for r in range(4))/math.sqrt(2)
                    score+=max(0,sum(a*b for a,b in zip(q,pooled))/math.sqrt(3))*weight
                scores.append(score)
            expected=[-1,-1,token if (token+1)%2 else -1]
            if scores:
                best=max(range(len(scores)),key=scores.__getitem__);expected[:2]=[2*best,2*best+1]
            assert selected[0,token].tolist()==expected
        h=torch.zeros(2,8,4,device='cuda');q=torch.zeros(2,8,2,device='cuda');k,g=gpu.project_keys(h)
        valid=torch.tensor([[False,False,True,True,True,True,True,False],[False]*8],device='cuda')
        positions=torch.arange(8,device='cuda').expand(2,-1)
        result=gpu.select(h,q,k,g,positions,valid_keys=valid)
        assert result[0].tolist()==[[-1,-1,-1],[-1,-1,-1],[-1,-1,2],[2,3,-1],[2,3,4],[2,3,-1],[2,3,6],[-1,-1,-1]]
        assert (result[1]==-1).all()
        for end in range(1,9):
            prefix=gpu.select(h[:,:end],q[:,:end],k[:,:end],g[:,:end],positions[:,:end],valid_keys=valid[:,:end])
            torch.testing.assert_close(prefix,result[:,:end],rtol=0,atol=0)
        holes=valid.clone();holes[0,3]=False
        rejected(lambda:gpu.select(h,q,k,g,positions,valid_keys=holes))
        rejected(lambda:gpu.select(h,q,k,g,positions.cpu()))
        rejected(lambda:gpu.select(h,q,k,g,positions,valid_keys=valid.cpu()))
        rejected(lambda:gpu.project_keys(hidden))
        rejected(lambda:FlashIndexer(weights,heads=2,pool_size=2,top_k=2))
        for option in ('max_workspace_bytes','max_output_bytes'):
            small=CudaFlashIndexer(weights,heads=2,pool_size=2,top_k=2,**{option:1})
            rejected(lambda:small.project_keys(h));rejected(lambda:small.select(h,q,k,g,positions))
        report.update(passed=True,scalar_ids_exact=True,prefix_causality=True,padding_ties_exact=True,invalid_inputs_rejected=True,
                      key_max_absolute=float((keys.cpu()-expected_keys).abs().max()),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
