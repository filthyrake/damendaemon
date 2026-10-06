"""Experimental adjacent Q4 kernel with runtime row bounds; no default change."""
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["M"])
def single_dot_dynamic_rows(X,W,S,BIAS,Y,M,N:tl.constexpr,K:tl.constexpr,
                          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BIASED:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    packed_columns=tl.arange(0,BK//2)
    groups=tl.arange(0,BK//32)
    accum=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+columns
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        p=start*(BK//2)+packed_columns
        packed=tl.load(W+outputs[:,None]*(K//2)+p[None,:],
                       (outputs[:,None]<N)&(p[None,:]<K//2),other=0).to(tl.int32)
        g=start*(BK//32)+groups
        scale=tl.load(S+outputs[:,None]*(K//32)+g[None,:],
                      (outputs[:,None]<N)&(g[None,:]<K//32),other=0).to(tl.float32)
        low=(((packed&15)-8).to(tl.float32).reshape(BN,BK//32,16)*scale[:,:,None]).to(tl.float16)
        high=(((packed>>4)-8).to(tl.float32).reshape(BN,BK//32,16)*scale[:,:,None]).to(tl.float16)
        w=tl.join(low,high).reshape(BN,BK)
        accum+=tl.dot(x,tl.trans(w))
    if BIASED:accum+=tl.load(BIAS+outputs,outputs<N,other=0)[None,:].to(tl.float32)
    tl.store(Y+rows[:,None]*N+outputs[None,:],accum,(rows[:,None]<M)&(outputs[None,:]<N))



def dynamic_rows_enabled(rows,width):
    # The down projection loses warm speed with partially filled 32/128-row
    # tiles. Keep those tile families specialized, based on the A2 sweep.
    return 1<=rows<=128 and (width==896 or (width==4864 and (rows<=16 or 32<rows<=64)))


def install_validation_dispatch():
    import os
    from dllm import q4_stage_kernels
    os.environ['DLLM_Q4_STAGE_ADJACENT']='1'
    original=q4_stage_kernels.single_dot_adjacent
    class Dispatch:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                kernel=single_dot_dynamic_rows if dynamic_rows_enabled(args[5],args[7]) else original
                return kernel[grid](*args,**kwargs)
            return launch
    q4_stage_kernels.single_dot_adjacent=Dispatch()
