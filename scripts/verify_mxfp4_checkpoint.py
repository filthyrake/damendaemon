"""Actual GPT-OSS expert tensors against independent CPU float64 products."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

from safetensors import safe_open
import torch
from damendaemon.mxfp4 import routed_linear
from verify_mxfp4 import reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    root=Path(args.model);source=json.loads((root/'damendaemon-source.json').read_text());catalog=source['tensors']
    torch.set_num_threads(4);torch.manual_seed(72);rows=[]
    def get(name):
        with safe_open(str(root/catalog[name]['file']),framework='pt',device='cpu') as reader:return reader.get_tensor(name)
    with torch.inference_mode():
        for layer in (0,11,23):
            for projection in ('gate_up_proj','down_proj'):
                name=f'model.layers.{layer}.mlp.experts.{projection}'
                picks=torch.tensor([0,31])
                blocks=get(name+'_blocks').index_select(0,picks).contiguous()
                scales=get(name+'_scales').index_select(0,picks).contiguous()
                bias=get(name+'_bias').index_select(0,picks).float().contiguous()
                weight=reference(blocks,scales)
                count=3;top=2;width=weight.shape[-1]
                selected=torch.tensor([[0,1],[1,0],[0,0]],dtype=torch.int64)
                shared=projection=='gate_up_proj'
                x=torch.randn(count if shared else count*top,width)
                target=torch.stack([weight[int(e)]@x[row//top if shared else row].double()+bias[int(e)].double()
                    for row,e in enumerate(selected.flatten())]).float()
                xb=x.cuda();wb=blocks.cuda();sb=scales.cuda();eb=selected.cuda();bb=bias.cuda()
                actual=routed_linear(xb,wb,sb,eb,bb,shared)
                torch.testing.assert_close(actual.cpu(),target,rtol=2e-5,atol=2e-4)
                error=(actual.cpu()-target).abs()
                rows.append({'layer':layer,'projection':projection,'checkpoint_experts':[0,31],
                    'scale_min':int(scales.min()),'scale_max':int(scales.max()),
                    'max_absolute_error':error.max().item(),
                    'relative_rms':(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-9)).item(),
                    'passed':True})
                del weight,xb,wb,sb,eb,bb,actual
        # A resident selected-expert FP32 baseline gets ideal preselected weights;
        # native routing remains on the GPU and reads the full packed expert bank.
        name='model.layers.0.mlp.experts.gate_up_proj'
        blocks=get(name+'_blocks');scales=get(name+'_scales');bias=get(name+'_bias').float()
        picks=torch.tensor([0,9,17,31]);selected=picks[None,:].cuda()
        decoded=reference(blocks.index_select(0,picks),scales.index_select(0,picks)).float().cuda()
        selected_bias=bias.index_select(0,picks).cuda().reshape(-1)
        x=torch.randn(1,decoded.shape[-1],device='cuda');wb=blocks.cuda();sb=scales.cuda();bb=bias.cuda()
        def native():return routed_linear(x,wb,sb,selected,bb,True)
        def baseline():return torch.nn.functional.linear(x,decoded.reshape(-1,decoded.shape[-1]),selected_bias)
        torch.testing.assert_close(native().reshape(-1),baseline().reshape(-1),rtol=2e-5,atol=2e-4)
        def measure(call):
            for _ in range(3):call()
            torch.cuda.synchronize();times=[]
            for _ in range(12):
                start=time.perf_counter();call();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
            return {'median_ms':statistics.median(times),'milliseconds':times}
        timings={'packed_native':measure(native),'preselected_fp32':measure(baseline),
            'packed_full_expert_bank_bytes':wb.numel()+sb.numel(),
            'expanded_only_four_experts_bytes':decoded.numel()*decoded.element_size(),
            'note':'One actual first-layer gate projection, one input and four routes. Baseline already has only selected FP32 expert weights resident; no model-generation speed claim.'}
    print(json.dumps({'passed':True,'repository':source['repository'],'revision':source['revision'],
        'gpu':torch.cuda.get_device_name(),'rows':rows,'timings':timings,
        'reference':'Independent CPU float64 format decoding and products on real checkpoint tensors'},indent=2))


if __name__=='__main__':main()
