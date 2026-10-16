"""Guarded tiny DeepSeek attention composition; no model/performance qualification."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_floating_attention import attention_prefill,attention_decode
    from dllm.floating_attention import attention_prefill as cpu_prefill
    from dllm.cuda_prepared_projection import CudaPackedProjection
    from tests.test_floating_attention import fixture
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def dense(query,kv,indices,sink,**kwargs):
        output=torch.zeros_like(query)
        for b in range(query.shape[0]):
            for t in range(query.shape[1]):
                vectors=kv[b,indices[b,t][indices[b,t]>=0]]
                for h in range(query.shape[2]):
                    logits=vectors@query[b,t,h]*query.shape[-1]**-.5
                    output[b,t,h]=torch.cat((logits,sink[h:h+1])).softmax(0)[:-1]@vectors
        return output
    def reference(hidden,weights,freq,ratio):
        with patch('dllm.floating_attention.selected_attention',side_effect=dense):
            output,state=cpu_prefill(hidden,weights,freq,groups=2,window=4,ratio=ratio,index_top_k=2,query_chunk=8)
        state.close();return output
    try:
        for packed in (False,True):
            for ratio,length,prefix in ((0,9,3),(4,9,3),(128,130,126)):
                hidden,weights,freq=fixture(ratio,length)
                device_weights={}
                names={'wq_a.weight','wq_b.weight','wkv.weight','wo_a.weight','wo_b.weight','indexer.wq_b.weight','indexer.weights_proj.weight'}
                for name,value in list(weights.items()):
                    if packed and name in names:
                        raw=value.to(torch.float8_e4m3fn).view(torch.uint8)
                        scales=torch.full(((value.shape[0]+127)//128,(value.shape[1]+127)//128),127,dtype=torch.uint8)
                        device_weights[name]=CudaPackedProjection(raw.cuda(),scales.cuda(),format='e4m3fn',row_chunk=16,column_chunk=16,workspace_bytes=65536)
                        weights[name]=raw.view(torch.float8_e4m3fn).float()
                    else:device_weights[name]=value.cuda()
                gpu=hidden.cuda();gpu_freq=freq.cuda()
                expected=reference(hidden[:,:prefix],weights,freq,ratio)
                actual,state=attention_prefill(gpu[:,:prefix],device_weights,gpu_freq,groups=2,window=4,ratio=ratio,index_top_k=2,query_chunk=8,max_tokens=length+2)
                try:
                    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)
                    error=float((actual.cpu()-expected).abs().max());pointer=state.cache.data_ptr()
                    for position in range(prefix,length):
                        actual=attention_decode(gpu[:,position:position+1],state,position=position)
                        expected=reference(hidden[:,:position+1],weights,freq,ratio)[:,-1:]
                        torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)
                        error=max(error,float((actual.cpu()-expected).abs().max()))
                        assert state.cache.data_ptr()==pointer
                    before=state.cache
                    try:state.reorder(torch.tensor([1],device='cuda'))
                    except ValueError:pass
                    else:raise AssertionError('out-of-range reorder admitted')
                    assert state.cache is before and not state.poisoned
                    state.reorder(torch.tensor([0,0],device='cuda'))
                    assert state.cache.shape[0]==2
                    actual=attention_decode(gpu[:,:1].expand(2,-1,-1),state,position=length)
                    expected=reference(torch.cat((hidden,hidden[:,:1]),dim=1),weights,freq,ratio)[:,-1:]
                    torch.testing.assert_close(actual.cpu(),expected.expand(2,-1,-1),rtol=3e-5,atol=3e-6)
                    state.reorder(torch.tensor([1],device='cuda'))
                    with patch.object(state.backend,'selected',side_effect=RuntimeError('injected late failure')):
                        try:attention_decode(gpu[:,:1],state,position=length+1)
                        except RuntimeError as exc:assert 'injected late failure' in str(exc)
                        else:raise AssertionError('failure not raised')
                    assert state.poisoned and state.position==length+1
                    report['cases'].append(dict(packed=packed,ratio=ratio,max_absolute=error,decode_steps=length-prefix,stable_cache=True,reorder=True,late_failure_poisoned=True))
                finally:state.close();state.close()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
