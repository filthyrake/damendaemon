"""Check prepared and resident weights for optional Metal Q8 decoders."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import json
import os
import torch
from damendaemon.models import load,weight_bytes
from damendaemon.runtime import require_gpu,make_executor
from damendaemon.device import empty_cache,synchronize


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    os.environ['DAMENDAEMON_DEVICE']='mps';os.environ['DAMENDAEMON_METAL_WEIGHT_DTYPE']='float16'
    require_gpu();torch.set_num_threads(4);reports=[]
    for quantization in ('none','q8_0'):
        model=load(args.model,512,quantization=quantization)
        estimate=weight_bytes(args.model,quantization=quantization)
        assert estimate==model.bytes,(estimate,model.bytes)
        executor=make_executor(model,max_batch=4)
        values=list(executor.w.values())
        if executor.quant_linear:values+=list(executor.quant_linear.scales.values())
        resident=sum(v.numel()*v.element_size() for v in {id(v):v for v in values}.values())
        reports.append({'quantization':quantization,'host_bytes':model.bytes,'preflight_bytes':estimate,
            'resident_weight_and_scale_bytes':resident,'packed_projections':len(model.scales),
            'embedding_dtype':str(executor.w['embed'].dtype),'head_dtype':str(executor.w['head'].dtype),
            'compute_dtype':str(executor.compute_dtype)})
        del values,executor,model
        gc.collect();synchronize();empty_cache()
    print(json.dumps({'passed':True,'model':args.model,'device':'mps','reports':reports},indent=2))


if __name__=='__main__':main()
