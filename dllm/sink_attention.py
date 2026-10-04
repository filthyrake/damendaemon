"""Bounded FP32 prefill attention with per-head softmax sinks.

The sink adds exp(sink) to the softmax denominator and zero to its numerator.
Rescale ordinary fused attention by sigmoid(logsumexp(scores) - sink). The
log-sum-exp output uses the efficient-attention operator in pinned PyTorch 2.6.
Only zero-offset, single-slot prefill calls use this path; decode retains the
native captured kernel.
"""
import torch


def prefill_attention(query,keys,values,sinks,window=0):
    if query.ndim!=4 or query.shape[2]!=1 or keys.ndim!=3 or values.ndim!=3:
        raise ValueError('fused sink attention requires [tokens,heads,1,dim] queries and [kv_heads,tokens,dim] caches')
    count,heads,_,dim=query.shape
    if (not count or not heads or not keys.shape[0] or query.dtype!=torch.float32 or keys.dtype!=query.dtype or values.dtype!=query.dtype
            or keys.shape!=values.shape or keys.shape[1:]!=(count,dim)
            or heads%keys.shape[0] or sinks.shape!=(heads,)):
        raise ValueError('fused sink attention requires FP32 single-prefix GQA tensors')
    group=heads//keys.shape[0]
    result=torch.empty_like(query)
    # Full attention partitions heads; sliding attention partitions query rows
    # and trims keys to the union of their legal windows. Both bound temporary
    # GQA expansion without constructing a context-squared mask.
    head_step=heads if window else min(heads,group)
    query_step=min(count,128) if window else count
    for first in range(0,count,query_step):
        last=min(count,first+query_step);left=max(0,first-window+1) if window else 0
        mask=None
        if window:
            qp=torch.arange(first,last,device=query.device)
            kp=torch.arange(left,last,device=query.device)
            valid=(kp[None,:]<=qp[:,None]) & (kp[None,:]>qp[:,None]-window)
            mask=torch.empty((last-first,((last-left+7)//8)*8),device=query.device,dtype=query.dtype)[:,:last-left]
            mask.fill_(0).masked_fill_(~valid,-float('inf'))
        for head in range(0,heads,head_step):
            stop=min(heads,head+head_step)
            q=query[first:last,head:stop,0].transpose(0,1)[None]
            k=keys[head//group:stop//group,left:last].repeat_interleave(group,dim=0)[None]
            v=values[head//group:stop//group,left:last].repeat_interleave(group,dim=0)[None]
            bias=None if mask is None else mask[None,None].expand(1,stop-head,last-first,last-left)
            output,lse,_,_=torch.ops.aten._scaled_dot_product_efficient_attention(
                q,k,v,bias,True,0.,not window,scale=dim**-.5)
            factor=torch.sigmoid(lse[:,:,:last-first]-sinks[None,head:stop,None])
            output.mul_(factor[:,:,:,None])
            result[first:last,head:stop,0].copy_(output[0].transpose(0,1))
            del q,k,v,output,lse,factor
    return result
