"""Check prepared Mamba accounting and retain which projections stay FP32."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import torch
from dllm.models import load,weight_bytes


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args();torch.set_num_threads(4)
    reports=[]
    for quantization in ('none','q8_0'):
        model=load(args.model,512,quantization=quantization)
        estimate=weight_bytes(args.model,quantization=quantization)
        assert estimate==model.bytes,(estimate,model.bytes)
        reports.append({'quantization':quantization,'bytes':model.bytes,'estimate':estimate,
            'packed_projections':len(model.scales),'embedding_dtype':str(model.weights['embed'].dtype),
            'dt_projection_dtype':str(model.weights['0.dt_proj'].dtype) if '0.dt_proj' in model.weights else None,
            'state_matrix_dtype':str(model.weights['0.A'].dtype)})
    print(json.dumps({'passed':True,'model':args.model,'reports':reports},indent=2))


if __name__=='__main__':main()
