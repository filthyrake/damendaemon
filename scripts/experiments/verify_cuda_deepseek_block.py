"""Guarded tiny CUDA DeepSeek blocks against decoded CPU controls."""
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
    from dllm.floating_block import FloatingBlock
    from dllm.cuda_deepseek_block import CudaDeepSeekBlock
    from dllm.cuda_prepared_projection import CudaPackedProjection
    from dllm.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
    from tests.test_floating_block import fixture
    from tests.test_packed_attention import NAMES
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def raw(value):
        data=value.to(torch.float8_e4m3fn).view(torch.uint8)
        scales=torch.full(((value.shape[0]+127)//128,(value.shape[1]+127)//128),127,dtype=torch.uint8)
        return data,scales,data.view(torch.float8_e4m3fn).float()
    try:
        for ratio,hashed in ((0,False),(4,False),(4,True)):
            block,hidden,ids=fixture(hashed,ratio)
            attention=dict(block.attention_weights);gpu_attention={}
            for name,value in attention.items():
                if name in NAMES:
                    data,scales,attention[name]=raw(value)
                    gpu_attention[name]=CudaPackedProjection(data.cuda(),scales.cuda(),format='e4m3fn',row_chunk=16,column_chunk=16,workspace_bytes=65536)
                else:gpu_attention[name]=value.cuda()
            weights=dict(block.weights);bank=[]
            for index in range(4):
                matrices=[]
                for part in ('w1','w3','w2'):
                    name=('ffn.experts.' if index<3 else 'ffn.shared_experts.')+part+'.weight'
                    value=weights[name][index] if index<3 else weights[name]
                    data,scales,decoded=raw(value)
                    if index<3:
                        if index==0:weights[name]=weights[name].clone()
                        weights[name][index]=decoded
                    else:weights[name]=decoded
                    matrices.append(CudaPackedMatrix(data.cuda(),scales.cuda(),'e4m3fn'))
                bank.append(CudaPackedExpert(*matrices,row_chunk=16,column_chunk=16,workspace_bytes=65536,activation_limit=10.))
            options=dict(groups=2,window=4,ratio=ratio,hc_mult=2,top_k=2,index_top_k=2)
            control=FloatingBlock(attention,weights,block.frequencies,**options)
            gpu_weights={k:v.cuda() for k,v in weights.items() if not k.startswith(('ffn.experts.','ffn.shared_experts.'))}
            candidate=CudaDeepSeekBlock(gpu_attention,gpu_weights,block.frequencies.cuda(),packed_experts=bank[:3],packed_shared=bank[3],**options)
            device=hidden.cuda();tokens=ids.cuda()
            expected,reference=control.prefill(hidden[:,:3],ids[:,:3],max_tokens=10,query_chunk=8)
            actual,state=candidate.prefill(device[:,:3],tokens[:,:3],max_tokens=10,query_chunk=8)
            try:
                torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                error=float((actual.cpu()-expected).abs().max())
                if hashed:
                    before=state.attention.cache.clone()
                    try:candidate.decode(device[:,3:4],state,torch.tensor([[99]],device='cuda'),position=3)
                    except ValueError:pass
                    else:raise AssertionError('invalid token admitted')
                    torch.testing.assert_close(state.attention.cache,before,rtol=0,atol=0,equal_nan=True)
                    assert not state.attention.poisoned and state.attention.position==3
                for position in range(3,7):
                    actual=candidate.decode(device[:,position:position+1],state,tokens[:,position:position+1],position=position)
                    expected=control.decode(hidden[:,position:position+1],reference,ids[:,position:position+1],position=position)
                    torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                    error=max(error,float((actual.cpu()-expected).abs().max()))
                state.reorder(torch.tensor([0,0],device='cuda'));reference.reorder(torch.tensor([0,0]))
                actual=candidate.decode(device[:,7:8].expand(2,-1,-1,-1),state,tokens[:,7:8].expand(2,-1),position=7)
                expected=control.decode(hidden[:,7:8].expand(2,-1,-1,-1),reference,ids[:,7:8].expand(2,-1),position=7)
                torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                with patch.object(candidate.backend,'ffn',side_effect=RuntimeError('injected late FFN failure')):
                    try:candidate.decode(device[:,8:9].expand(2,-1,-1,-1),state,tokens[:,8:9].expand(2,-1),position=8)
                    except RuntimeError as exc:assert 'injected late FFN failure' in str(exc)
                    else:raise AssertionError('late failure not raised')
                assert state.attention.poisoned and state.attention.position==9
                report['cases'].append(dict(ratio=ratio,hashed=hashed,max_absolute=error,reorder=True,late_failure_poisoned=True))
            finally:state.close();state.close();reference.close()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
