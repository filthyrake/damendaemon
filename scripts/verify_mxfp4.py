"""A2 MXFP4 byte semantics, routed projections and graph replay proof."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import os

import torch
from damendaemon.mxfp4 import unpack,routed_linear


# Independent reference uses the published E2M1 value table and float64 ldexp.
def reference(blocks,scales):
    table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    b=blocks.cpu();s=scales.cpu()
    codes=torch.stack((b & 15,b >> 4),dim=-1).reshape(*b.shape[:-1],32).long()
    result=torch.ldexp(table[codes],(s.int()-127)[...,None])
    result=torch.where((s==255)[...,None],float('nan'),result)
    return result.reshape(*b.shape[:-2],b.shape[-2]*32)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);torch.manual_seed(39);rows=[]
    with torch.inference_mode():
        scales=torch.arange(256,dtype=torch.uint8).view(1,-1)
        codes=torch.arange(16,dtype=torch.uint8).repeat(2)
        packed=(codes[::2] | (codes[1::2]<<4)).repeat(1,scales.numel(),1).contiguous()
        expected=reference(packed,scales)
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            actual=unpack(packed.cuda(),scales.cuda(),dtype).cpu()
            target=expected.to(dtype)
            torch.testing.assert_close(actual,target,rtol=0,atol=0,equal_nan=True)
            zeros=(target==0)
            assert torch.equal(torch.signbit(actual[zeros]),torch.signbit(target[zeros]))
            rows.append({'case':'all_codes_all_256_scales','dtype':str(dtype),'exact_values':True,'signed_zero':True})
        # Exact FP16 expansion must also survive the compensated matrix path,
        # including the lowest subnormal and largest safe expert scales.
        from damendaemon.precise_matmul import linear as precise_linear
        for exponent in (104,105,140):
            blocks=torch.randint(0,256,(37,3,16),dtype=torch.uint8)
            scales=torch.full((37,3),exponent,dtype=torch.uint8)
            inputs=torch.randn(17,96)
            target=(inputs.double()@reference(blocks,scales).t()).float()
            actual=precise_linear(inputs.cuda(),unpack(blocks.cuda(),scales.cuda(),torch.float16)).cpu()
            error=(actual-target).abs()
            relative=(error.square().mean().sqrt()/target.square().mean().sqrt()).item()
            assert relative<2e-6 and error.max().item()<1e-5*max(1e-12,target.abs().max().item())
            rows.append({'case':'fp16_expansion_compensated_matrix','scale':exponent,'relative_rms':relative})
        for count,experts,outputs,width,top in ((1,5,37,96,3),(3,5,37,96,3),(4,4,257,2880,2)):
            blocks=torch.randint(0,256,(experts,outputs,width//32,16),dtype=torch.uint8)
            scales=torch.randint(119,125,(experts,outputs,width//32),dtype=torch.uint8)
            weights=reference(blocks,scales)
            selected=torch.randint(experts,(count,top),dtype=torch.int64)
            bias=torch.randn(experts,outputs,dtype=torch.float32)*.02
            for shared in (True,False):
                x=torch.randn(count if shared else count*top,width)
                target=torch.stack([weights[int(e)]@x[row//top if shared else row].double()+bias[int(e)].double()
                    for row,e in enumerate(selected.flatten())]).float()
                actual=routed_linear(x.cuda(),blocks.cuda(),scales.cuda(),selected.cuda(),bias.cuda(),shared).cpu()
                torch.testing.assert_close(actual,target,rtol=2e-5,atol=3e-5)
                rows.append({'case':'routed','tokens':count,'experts':experts,'outputs':outputs,'width':width,
                    'top':top,'shared_input':shared,'max_absolute_error':(actual-target).abs().max().item()})
        from damendaemon.mxfp4_kernels import swiglu
        activation=torch.randn(3,5760,device='cuda')*20
        activation[0,:10]=torch.tensor([-1000,1000,-7,7,0,-1,7,-7,1000,-1000],device='cuda')
        gate=activation.double()[:,::2].clamp(max=7)
        up=activation.double()[:,1::2].clamp(-7,7)
        target_activation=(gate*torch.sigmoid(1.702*gate)*(up+1)).float()
        actual_activation=swiglu(activation)
        torch.testing.assert_close(actual_activation,target_activation,rtol=3e-6,atol=2e-6)
        rows.append({'case':'interleaved_clipped_swiglu','max_absolute_error':
            (actual_activation-target_activation).abs().max().item()})
        # Graph capture must follow live device-selected experts, not freeze IDs.
        xb=x.cuda();wb=blocks.cuda();sb=scales.cuda();eb=selected.cuda();bb=bias.cuda()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):routed_linear(xb,wb,sb,eb,bb,False)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=routed_linear(xb,wb,sb,eb,bb,False)
        for offset in (1,2):
            changed=(selected+offset)%experts;eb.copy_(changed.cuda());graph.replay()
            target=torch.stack([weights[int(e)]@x[row].double()+bias[int(e)].double()
                for row,e in enumerate(changed.flatten())]).float()
            torch.testing.assert_close(captured.cpu(),target,rtol=2e-5,atol=3e-5)
        bad=selected.clone();bad[0,0]=-1;bad[0,1]=experts
        invalid=routed_linear(xb,wb,sb,bad.cuda(),bb,False)
        assert torch.isnan(invalid[:2]).all()
        bad[0,0]=1<<32;bad[0,1]=-(1<<32)
        invalid=routed_linear(xb,wb,sb,bad.cuda(),bb,False)
        assert torch.isnan(invalid[:2]).all()
        rows.append({'case':'graph_and_invalid_route_bounds','live_expert_replay':True,'invalid_routes_nan':True})
    print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'rows':rows,'reference':'Independent CPU float64 E2M1 table/ldexp and matrix products; no inference runtime'},indent=2))


if __name__=='__main__':main()
