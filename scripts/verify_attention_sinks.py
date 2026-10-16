"""Independent A2 checks for one global attention-sink normalization term."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import os

import torch
from dllm import kernels


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);torch.manual_seed(83);rows=[]
    with torch.inference_mode():
        for context in (128,257,1024):
            heads,kv_heads,dim=8,2,64
            cache_k=torch.randn(4,kv_heads,context,dim,device='cuda')
            cache_v=torch.randn_like(cache_k)
            slots=torch.tensor([3,1],device='cuda');positions=torch.tensor([context-1,17],device='cuda')
            query=torch.randn(2,heads,1,dim,device='cuda')
            for window in (0,33,128):
                for mode in ('none','negative','mixed','positive'):
                    sink=None if mode=='none' else (torch.linspace(-5,5,heads,device='cuda') if mode=='mixed'
                        else torch.full((heads,),100. if mode=='positive' else -100.,device='cuda'))
                    actual=kernels.attention(query,cache_k,cache_v,positions,slots,window=window,sinks=sink)
                    key=cache_k[slots].repeat_interleave(heads//kv_heads,1)
                    value=cache_v[slots].repeat_interleave(heads//kv_heads,1)
                    logits=(query.double()@key.double().transpose(-1,-2))*(dim**-.5)
                    times=torch.arange(context,device='cuda')
                    mask=times[None,:]<=positions[:,None]
                    if window:mask=mask & (times[None,:]>positions[:,None]-window)
                    logits=logits.masked_fill(~mask[:,None,None,:],float('-inf'))
                    if sink is not None:logits=torch.cat((logits,sink.double()[None,:,None,None].expand(2,-1,1,-1)),dim=-1)
                    probability=logits.softmax(-1)[...,:context]
                    expected=(probability@value.double()).float()
                    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
                    rows.append({'context':context,'window':window,'sink_mode':mode,
                        'max_absolute_error':(actual-expected).abs().max().item(),'passed':True})
        # Sinks remain live inputs during replay, including split reduction.
        sink=torch.zeros(heads,device='cuda')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):kernels.attention(query,cache_k,cache_v,positions,slots,sinks=sink)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=kernels.attention(query,cache_k,cache_v,positions,slots,sinks=sink)
        sink.fill_(100);graph.replay();assert out.abs().max().item()<1e-30
        sink.fill_(-100);graph.replay()
        plain=kernels.attention(query,cache_k,cache_v,positions,slots)
        torch.testing.assert_close(out,plain,rtol=2e-5,atol=3e-6)
    print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'rows':rows,'live_sink_graph_replay':True,
        'reference':'Independent FP64 QK/softmax including one sink logit and a zero value vector'},indent=2))


if __name__=='__main__':main()
