"""Native CUDA block decoding for GGUF K-quants and mixed projection rows.

Format reference: MIT-licensed gguf-py/gguf/quants.py in llama.cpp,
836d57176dc699a726c55418e4f96b8ca628e1bf. Independently implemented GPU arithmetic.
Packed coefficients are decoded directly by CUDA kernels.
"""
import os

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
def _kblock_values(W, rows, columns, K:tl.constexpr, KIND:tl.constexpr,
                   OFFSET:tl.constexpr, selected):
    # Read each block/group header once, then broadcast it over its coefficients.
    # Elementwise header loads otherwise repeat the same decode 16-32 times.
    BK:tl.constexpr=columns.shape[0]
    blocks=tl.min(columns,0)//256+tl.arange(0,BK//256)
    SIZE:tl.constexpr=144 if KIND==12 else 176 if KIND==13 else 210
    base=OFFSET+(rows[:,None]*(K//256)+blocks[None,:])*SIZE
    valid=selected[:,None]&(blocks[None,:]<K//256)
    if KIND==12 or KIND==13:
        group=tl.arange(0,8)
        lane=tl.arange(0,32)
        d=tl.load((W+base).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        dm=tl.load((W+base+2).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        index=group%4
        sc=tl.load(W+base[:,:,None]+4+index[None,None,:],valid[:,:,None],other=0).to(tl.int32)
        mn=tl.load(W+base[:,:,None]+8+index[None,None,:],valid[:,:,None],other=0).to(tl.int32)
        shared=tl.load(W+base[:,:,None]+12+index[None,None,:],valid[:,:,None],other=0).to(tl.int32)
        scale=tl.where(group[None,None,:]<4,sc&63,(shared&15)|((sc>>2)&48))
        minimum=tl.where(group[None,None,:]<4,mn&63,(shared>>4)|((mn>>2)&48))
        ds=d[:,:,None]*scale.to(tl.float32)
        ms=dm[:,:,None]*minimum.to(tl.float32)
        QUANT:tl.constexpr=16 if KIND==12 else 48
        bits=tl.load(W+base[:,:,None,None]+QUANT+(group[None,None,:,None]//2)*32+lane[None,None,None,:],
                     valid[:,:,None,None],other=0).to(tl.int32)
        q=(bits>>((group[None,None,:,None]%2)*4))&15
        if KIND==13:
            high=tl.load(W+base[:,:,None]+16+lane[None,None,:],valid[:,:,None],other=0).to(tl.int32)
            q=q|(((high[:,:,None,:]>>group[None,None,:,None])&1)<<4)
        values=ds[:,:,:,None]*q.to(tl.float32)-ms[:,:,:,None]
    else:
        group=tl.arange(0,16)
        lane=tl.arange(0,16)
        pos=group[:,None]*16+lane[None,:]
        low=tl.load(W+base[:,:,None,None]+(pos[None,None,:,:]//128)*64+pos[None,None,:,:]%64,
                    valid[:,:,None,None],other=0).to(tl.int32)
        high=tl.load(W+base[:,:,None,None]+128+(pos[None,None,:,:]//128)*32+pos[None,None,:,:]%32,
                     valid[:,:,None,None],other=0).to(tl.int32)
        q=((low>>((pos[None,None,:,:]%128//64)*4))&15)|(((high>>((pos[None,None,:,:]%128//32)*2))&3)<<4)
        scale=tl.load(W+base[:,:,None]+192+group[None,None,:],valid[:,:,None],other=0).to(tl.int8).to(tl.float32)
        d=tl.load((W+base+208).to(tl.pointer_type(tl.float16)),valid,other=0).to(tl.float32)
        ds=d[:,:,None]*scale
        values=ds[:,:,:,None]*(q-32).to(tl.float32)
    return values.reshape(rows.shape[0],BK)


@tr.jit
def _segment_values(W, rows, columns, K:tl.constexpr, SPEC:tl.constexpr, SEGMENT:tl.constexpr):
    first:tl.constexpr=SPEC.value[SEGMENT][0]
    last:tl.constexpr=SPEC.value[SEGMENT][1]
    kind:tl.constexpr=SPEC.value[SEGMENT][2]
    offset:tl.constexpr=SPEC.value[SEGMENT][3]
    selected=(rows>=first)&(rows<last)
    valid=selected[:,None]&(columns[None,:]<K)
    if kind>=12 and columns.shape[0]>=256:
        values=_kblock_values(W,rows-first,columns,K,kind,offset,selected)
    else:
        values=_block_values(W,rows-first,columns,K,kind,offset,valid)
    return tl.where(valid,values,0)


@tr.jit
def _values(W, rows, columns, K:tl.constexpr, SPEC:tl.constexpr):
    result=tl.full((rows.shape[0],columns.shape[0]),0,tl.float32)
    for segment in tl.static_range(len(SPEC)):
        result+=_segment_values(W,rows,columns,K,SPEC,segment)
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
          BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr,PARTS:tl.constexpr=1):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    part=tl.program_id(1)
    cols=part*BK+tl.arange(0,BK)
    values=_values(W,rows,cols,K,SPEC).to(X.dtype.element_ty).to(tl.float32)
    for m in range(M):
        x=tl.load(X+m*K+cols,cols<K,other=0).to(tl.float32)
        result=tl.sum(values*x[None,:],1)
        if BIASED and PARTS==1:result+=tl.load(BIAS+rows,rows<N,other=0).to(tl.float32)
        tl.store(Y+(m*N+rows)*PARTS+part,result,rows<N)


@tr.jit
def _reduce(P,BIAS,Y,N:tl.constexpr,M:tl.constexpr,PARTS:tl.constexpr,
            BIASED:tl.constexpr,B:tl.constexpr):
    offsets=tl.program_id(0)*B+tl.arange(0,B)
    parts=tl.arange(0,tr.next_power_of_2(PARTS))
    values=tl.load(P+offsets[:,None]*PARTS+parts[None,:],
                   (offsets[:,None]<M*N)&(parts[None,:]<PARTS),other=0)
    result=tl.sum(values,1)
    if BIASED:result+=tl.load(BIAS+offsets%N,offsets<M*N,other=0).to(tl.float32)
    tl.store(Y+offsets,result,offsets<M*N)


def split_gemv(x,weight,layout,bias=None,*,bn=4,bk=512,warps=4):
    width,spec=layout;rows=x.shape[0];outputs=spec[-1][1];parts=tr.cdiv(width,bk)
    result=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
    partial=torch.empty((rows,outputs,parts),device=x.device,dtype=torch.float32)
    _gemv[(tr.cdiv(outputs,bn),parts)](x,weight,weight if bias is None else bias,partial,
        outputs,width,rows,spec,bn,bk,False,parts,enable_fp_fusion=False,num_warps=warps)
    _reduce[(tr.cdiv(rows*outputs,128),)](partial,weight if bias is None else bias,result,
        outputs,rows,parts,bias is not None,128,enable_fp_fusion=False)
    return result


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
    if rows<=4:
        from .native_kquant import linear_if_enabled
        native=linear_if_enabled(x,weight,layout,bias)
        if native is not None:return native.reshape(*shape,outputs)
    if rows<=4 and width>=4096:
        result=split_gemv(x,weight,layout,bias,bn=8,bk=1024)
    elif rows<=4:
        result=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
        _gemv[(tr.cdiv(outputs,4),)](x,weight,weight if bias is None else bias,result,outputs,width,rows,
            spec,4,tr.next_power_of_2(width),bias is not None,enable_fp_fusion=False)
    elif x.dtype == torch.float16 and os.environ.get('DAMENDAEMON_TRITON_PREFILL_GEMM') == '1':
        # Shape-invariant opt-in prefill GEMM (#585): the per-row K-loop of
        # _gemm is independent of the row count, so 128-row chunked calls and
        # a whole 384-row prefill produce bit-identical rows. The default
        # expand+cuBLAS path below is measured faster but selects cuBLAS
        # kernels per (M, N, K); at qwen3-8B the out (4096x4096) and down
        # (4096x6144) projections round differently between M=384 and
        # M=128 (1.4-2.2e-5 per element), which chunked-prefill comparisons
        # then amplify through decode near-ties. FP16 packed prefill only;
        # the FP32 policy paths do not consult this variable. Opt in when
        # GEMM shape-invariance matters more than speed.
        result=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
        _gemm[(tr.cdiv(rows,64),tr.cdiv(outputs,64))](x,weight,
            weight if bias is None else bias,result,rows,outputs,width,spec,
            64,64,64,bias is not None,enable_fp_fusion=False)
    else:
        # Measured faster than repeated packed Tensor Core tile decoding on A2.
        # Only this projection is expanded; persistent weights remain packed.
        result=torch.nn.functional.linear(x,expand(weight,layout,x.dtype),bias)
    return result.reshape(*shape,outputs)
