"""Native CUDA block decoding for GGUF K-quants and mixed projection rows.

Format reference: MIT-licensed gguf-py/gguf/quants.py in llama.cpp,
836d57176dc699a726c55418e4f96b8ca628e1bf. Independently implemented GPU arithmetic.
This module is not yet connected to model loading; qualification precedes dispatch.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _block_values(W, rows, columns, K:tl.constexpr, KIND:tl.constexpr, OFFSET:tl.constexpr, valid):
    r, c = rows[:,None], columns[None,:]
    if KIND == 1:
        return tl.load((W + OFFSET).to(tl.pointer_type(tl.float16)) + r*K+c, valid, other=0).to(tl.float32)
    if KIND == 2 or KIND == 8:
        SIZE:tl.constexpr = 18 if KIND == 2 else 34
        base = OFFSET+(r*(K//32)+c//32)*SIZE
        d = tl.load((W+base).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        if KIND == 2:
            bits=tl.load(W+base+2+c%16,valid,other=0).to(tl.int32)
            q=((bits>>((c%32//16)*4))&15)-8
        else:
            q=tl.load(W+base+2+c%32,valid,other=0).to(tl.int8).to(tl.int32)
        return d*q.to(tl.float32)
    if KIND == 12 or KIND == 13:
        SIZE:tl.constexpr = 144 if KIND == 12 else 176
        base=OFFSET+(r*(K//256)+c//256)*SIZE
        pos=c%256; group=pos//32; index=group%4
        d=tl.load((W+base).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        dm=tl.load((W+base+2).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        sc=tl.load(W+base+4+index,valid,other=0).to(tl.int32)
        mn=tl.load(W+base+8+index,valid,other=0).to(tl.int32)
        shared=tl.load(W+base+12+index,valid,other=0).to(tl.int32)
        scale=tl.where(group<4,sc&63,(shared&15)|((sc>>2)&48))
        minimum=tl.where(group<4,mn&63,(shared>>4)|((mn>>2)&48))
        QUANT_OFFSET:tl.constexpr = 16 if KIND == 12 else 48
        bits=tl.load(W+base+QUANT_OFFSET+(pos//64)*32+pos%32,valid,other=0).to(tl.int32)
        q=(bits>>((group%2)*4))&15
        if KIND == 13:
            high=tl.load(W+base+16+pos%32,valid,other=0).to(tl.int32)
            q=q|(((high>>group)&1)<<4)
        return (d*scale.to(tl.float32))*q.to(tl.float32)-dm*minimum.to(tl.float32)
    if KIND == 14:
        base=OFFSET+(r*(K//256)+c//256)*210
        pos=c%256; half=pos//128
        low=tl.load(W+base+half*64+pos%64,valid,other=0).to(tl.int32)
        high=tl.load(W+base+128+half*32+pos%32,valid,other=0).to(tl.int32)
        q=((low>>((pos%128//64)*4))&15)|(((high>>((pos%128//32)*2))&3)<<4)
        scale=tl.load(W+base+192+pos//16,valid,other=0).to(tl.int8).to(tl.float32)
        d=tl.load((W+base+208).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        return (d*scale)*(q-32).to(tl.float32)


@tr.jit
def _values(W, rows, columns, K:tl.constexpr, SPEC:tl.constexpr):
    result=tl.full((rows.shape[0],columns.shape[0]),0,tl.float32)
    for segment in tl.static_range(len(SPEC)):
        first=SPEC.value[segment][0]
        last=SPEC.value[segment][1]
        kind=SPEC.value[segment][2]
        offset=SPEC.value[segment][3]
        selected=(rows>=first)&(rows<last)
        valid=selected[:,None]&(columns[None,:]<K)
        values=_block_values(W,rows-first,columns,K,kind,offset,valid)
        result+=tl.where(valid,values,0)
    return result


@tr.jit
def _expand(W,Y,N:tl.constexpr,K:tl.constexpr,SPEC:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    rows=tl.program_id(0)*BR+tl.arange(0,BR)
    cols=tl.program_id(1)*BC+tl.arange(0,BC)
    values=_values(W,rows,cols,K,SPEC)
    tl.store(Y+rows[:,None]*K+cols[None,:],values,(rows[:,None]<N)&(cols[None,:]<K))


def expand(weight,layout,dtype):
    width,spec=layout; rows=spec[-1][1]
    result=torch.empty((rows,width),device=weight.device,dtype=dtype)
    _expand[(tr.cdiv(rows,4),tr.cdiv(width,256))](weight,result,rows,width,spec,4,256,enable_fp_fusion=False)
    return result


@tr.jit
def _embedding(IDS,W,Y,M:tl.constexpr,K:tl.constexpr,SPEC:tl.constexpr,B:tl.constexpr):
    position=tl.program_id(0)
    rows=tl.load(IDS+position+tl.arange(0,1))
    cols=tl.program_id(1)*B+tl.arange(0,B)
    values=_values(W,rows,cols,K,SPEC).to(tl.float16).to(Y.dtype.element_ty)
    tl.store(Y+position*K+cols[None,:],values,cols[None,:]<K)


def embedding(ids,weight,layout,dtype):
    width,spec=layout; ids=ids.contiguous()
    result=torch.empty((*ids.shape,width),device=weight.device,dtype=dtype)
    if ids.numel():
        _embedding[(ids.numel(),tr.cdiv(width,256))](ids,weight,result,ids.numel(),width,spec,256,enable_fp_fusion=False)
    return result


@tr.jit
def _gemv(X,W,BIAS,Y,N:tl.constexpr,K:tl.constexpr,M:tl.constexpr,SPEC:tl.constexpr,
          BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    cols=tl.arange(0,BK)
    values=_values(W,rows,cols,K,SPEC).to(X.dtype.element_ty).to(tl.float32)
    for m in range(M):
        x=tl.load(X+m*K+cols,cols<K,other=0).to(tl.float32)
        result=tl.sum(values*x[None,:],1)
        if BIASED:result+=tl.load(BIAS+rows,rows<N,other=0).to(tl.float32)
        tl.store(Y+m*N+rows,result,rows<N)


@tr.jit
def _gemm(X,W,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,SPEC:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    block=tl.arange(0,BK)
    result=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        columns=start*BK+block
        x=tl.load(X+rows[:,None]*K+columns[None,:],(rows[:,None]<M)&(columns[None,:]<K),other=0)
        weight=_values(W,outputs,columns,K,SPEC).to(tl.float16)
        result+=tl.dot(x,tl.trans(weight))
    if BIASED:result+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],result,(rows[:,None]<M)&(outputs[None,:]<N))


def linear(x,weight,layout,bias=None):
    shape=x.shape[:-1]; x=x.reshape(-1,x.shape[-1]).contiguous()
    width,spec=layout; rows=x.shape[0]; outputs=spec[-1][1]
    if x.shape[1]!=width:raise ValueError('packed matrix width mismatch')
    if x.dtype==torch.float16:
        result=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
        _gemm[(tr.cdiv(rows,16),tr.cdiv(outputs,32))](x,weight,weight if bias is None else bias,result,
            rows,outputs,width,spec,16,32,128,bias is not None,enable_fp_fusion=False,num_stages=2)
    elif rows<=4:
        result=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
        _gemv[(tr.cdiv(outputs,4),)](x,weight,weight if bias is None else bias,result,outputs,width,rows,
            spec,4,tr.next_power_of_2(width),bias is not None,enable_fp_fusion=False)
    else:
        result=torch.nn.functional.linear(x,expand(weight,layout,x.dtype),bias)
    return result.reshape(*shape,outputs)
