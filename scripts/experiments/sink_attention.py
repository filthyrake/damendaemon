"""Experimental fused FP32 attention with sink mass recovered from log-sum-exp.

The sink contributes exp(sink) to the softmax denominator and zero to its
numerator. A fused ordinary attention result can therefore be rescaled by
sigmoid(logsumexp(scores) - sink), without materializing per-head score matrices.
This uses a version-specific PyTorch operator and is not a serving backend.
"""
import torch


def attention(query,keys,values,sinks,window=0):
    # Input: [tokens, heads, 1, dim], [kv_heads, tokens, dim]. This experiment
    # handles an ordinary zero-offset, single-slot prefill only.
    count,heads,_,dim=query.shape
    q=query[:, :, 0].transpose(0,1)[None]
    repeats=heads//keys.shape[0]
    k=keys.repeat_interleave(repeats,dim=0)[None]
    v=values.repeat_interleave(repeats,dim=0)[None]
    mask=None
    if window:
        positions=torch.arange(count,device=query.device)
        valid=(positions[None,:]<=positions[:,None]) & (positions[None,:]>positions[:,None]-window)
        # The efficient backend requires an aligned bias row stride.
        mask=torch.empty((count,((count+7)//8)*8),device=query.device,dtype=query.dtype)[:,:count]
        mask.fill_(0).masked_fill_(~valid,-float('inf'))
        mask=mask[None,None].expand(1,heads,count,count)
    output,lse,_,_=torch.ops.aten._scaled_dot_product_efficient_attention(
        q,k,v,mask,True,0.,not window,scale=dim**-.5)
    factor=torch.sigmoid(lse[:,:,:count]-sinks[None,:,None])
    output=output*factor[:,:,:,None]
    return output[0].transpose(0,1)[:,:,None,:].contiguous()


def bounded_attention(query,keys,values,sinks,window=0):
    count,heads,_,dim=query.shape;group=heads//keys.shape[0]
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
