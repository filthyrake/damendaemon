"""Compare fused packed prefill projections with unpack-then-matmul, including unpack cost."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import statistics
import torch
from damendaemon.qwen2 import ModelWeights
from damendaemon.mxfp4 import unpack,scale_capabilities
from damendaemon.precise_matmul import linear as dense
from scripts.experiments.mxfp4_prefill import linear as packed


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.manual_seed(47);model=ModelWeights(args.model,512);rows=[]
    with torch.inference_mode():
        for layer in (0,23):
            for projection in ('up','down'):
                blocks=model.weights[f'{layer}.{projection}.blocks'][0].cuda()
                scales=model.weights[f'{layer}.{projection}.scales'][0].cuda()
                assert scale_capabilities(scales)[0]
                width=blocks.shape[-2]*32
                for count in (16,128,512):
                    x=torch.randn(count,width,device='cuda')*(1 if projection=='up' else 5)
                    def baseline():return dense(x,unpack(blocks,scales,torch.float16))
                    def candidate():return packed(x,blocks,scales)
                    expected=baseline();actual=candidate();exact=torch.equal(actual,expected)
                    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-5)
                    # Independent selected-row/column FP64 product over the
                    # already independently qualified exact unpack operation.
                    r=torch.tensor([0,count//2,count-1],device='cuda');c=torch.arange(0,blocks.shape[0],max(1,blocks.shape[0]//64),device='cuda')
                    weights=unpack(blocks,scales,torch.float32).index_select(0,c).cpu().double()
                    truth=x.index_select(0,r).cpu().double()@weights.t()
                    selected=actual.index_select(0,r).index_select(1,c).cpu().double()
                    rms=float((selected-truth).square().mean().sqrt()/truth.square().mean().sqrt())
                    assert rms<3e-6
                    samples={'baseline':[],'candidate':[]};peaks={}
                    for round_index in range(8):
                        modes=[('baseline',baseline),('candidate',candidate)]
                        if round_index%2:modes.reverse()
                        for name,fn in modes:
                            torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            start.record();result=fn();end.record();end.synchronize()
                            if round_index>=2:samples[name].append(start.elapsed_time(end))
                            peaks[name]=torch.cuda.max_memory_allocated()-before;del result
                    row={'layer':layer,'projection':projection,'tokens':count,'bit_exact':exact,'max_abs':float((actual-expected).abs().max()),
                        'selected_fp64_relative_rms':rms,'milliseconds':{k:statistics.median(v) for k,v in samples.items()},
                        'samples':samples,'incremental_peak_bytes':peaks}
                    rows.append(row);print(json.dumps(row),flush=True)
                    Path(args.output).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2)+'\n')
                    del x,actual,expected
                del blocks,scales


if __name__=='__main__':main()
