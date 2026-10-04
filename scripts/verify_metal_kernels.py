"""Native Metal kernel checks against independent CPU FP32 tensor operations."""
import json
import torch
from torch.nn import functional as F
from dllm import metal_kernels as metal


@torch.inference_mode()
def main():
    torch.manual_seed(1829)
    checks = []
    for rows in (1, 3, 17):
        for dim in (64, 1024, 1536):
            x = torch.randn(rows,dim)
            w = torch.randn(dim)
            target = x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1.e-6)*w
            torch.testing.assert_close(metal.rms(x.to('mps'),w.to('mps'),1.e-6).cpu(),target,atol=2.e-5,rtol=2.e-5)
            gate,up=x.chunk(2,-1)
            torch.testing.assert_close(metal.swiglu(x.to('mps')).cpu(),F.silu(gate)*up,atol=2.e-5,rtol=2.e-5)
            checks.append({'op':'norm_and_activation','rows':rows,'dim':dim})
    for rows in (1,2,3,4):
        x=torch.randn(rows,1024);w=torch.randn(129,1024);b=torch.randn(129)
        for bias in (None,b):
            actual=metal.linear(x.to('mps'),w.to('mps'),None if bias is None else bias.to('mps')).cpu()
            torch.testing.assert_close(actual,F.linear(x,w,bias),atol=1.e-4,rtol=1.e-4)
        checks.append({'op':'linear','rows':rows})
    for dim in (64,96,128):
        h,kh,context=8,2,512
        slots=torch.tensor([3,0,2]);positions=torch.tensor([5,31,257])
        qkv=torch.randn(3,(h+2*kh)*dim)
        phases=torch.randn(context,dim)
        cos,sin=phases.cos(),phases.sin()
        qw,kw=torch.randn(dim),torch.randn(dim)
        for normalize in (False,True):
            ck,cv=[torch.randn(4,kh,context,dim) for _ in range(2)]
            k_gpu,v_gpu=ck.to('mps'),cv.to('mps')
            q=metal.rope_batch(qkv.to('mps'),cos.to('mps'),sin.to('mps'),positions.to('mps'),slots.to('mps'),
                k_gpu,v_gpu,h,qw.to('mps') if normalize else None,kw.to('mps') if normalize else None)
            query,key,value=qkv.split((h*dim,kh*dim,kh*dim),-1)
            query,key,value=query.view(3,h,1,dim),key.view(3,kh,1,dim),value.view(3,kh,dim)
            if normalize:
                query=query*torch.rsqrt(query.square().mean(-1,keepdim=True)+1.e-6)*qw
                key=key*torch.rsqrt(key.square().mean(-1,keepdim=True)+1.e-6)*kw
            def rotate(x):
                a,b=x.chunk(2,-1)
                return x*cos[positions,None,None]+torch.cat((-b,a),-1)*sin[positions,None,None]
            query,key=rotate(query),rotate(key)
            ck[slots,:,positions]=key.squeeze(2);cv[slots,:,positions]=value
            torch.testing.assert_close(q.cpu(),query,atol=2.e-5,rtol=2.e-5)
            torch.testing.assert_close(k_gpu.cpu(),ck,atol=2.e-5,rtol=2.e-5)
            torch.testing.assert_close(v_gpu.cpu(),cv,atol=0,rtol=0)
            actual=metal.attention(q,k_gpu,v_gpu,positions.to('mps'),slots.to('mps')).cpu()
            mask=torch.arange(context)[None,None,None,:] <= positions[:,None,None,None]
            target=F.scaled_dot_product_attention(query,ck[slots],cv[slots],attn_mask=mask,enable_gqa=True)
            torch.testing.assert_close(actual,target,atol=3.e-5,rtol=3.e-5)
            checks.append({'op':'rope_cache_attention','dim':dim,'qk_norm':normalize})
    # Cover the native shared-memory boundary and the larger-context GPU fallback.
    for context in (2048,4096,4097):
        for single in (False,True):
            slots=torch.tensor([0] if single else [3,0,2])
            positions=torch.tensor([context-1] if single else [context-1,1023,31])
            q=torch.randn(len(slots),8,1,64)
            k,v=[torch.randn(1 if single else 4,2,context,64) for _ in range(2)]
            actual=metal.attention(q.to('mps'),k.to('mps'),v.to('mps'),positions.to('mps'),
                                   None if single else slots.to('mps')).cpu()
            mask=torch.arange(context)[None,None,None,:] <= positions[:,None,None,None]
            target=F.scaled_dot_product_attention(q,k[slots],v[slots],attn_mask=mask,enable_gqa=True)
            torch.testing.assert_close(actual,target,atol=3.e-5,rtol=3.e-5)
            checks.append({'op':'attention_boundary','context':context,'single':single})
    print(json.dumps({'passed':True,'torch':str(torch.__version__),'checks':checks},indent=2))


if __name__ == '__main__':
    main()
