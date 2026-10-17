"""Ablate projection arithmetic against independent routing-sensitive outputs."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from damendaemon.qwen2 import ModelWeights
from damendaemon.gpt_oss import GptOssExecutor
from scripts.experiments.precise_projection import linear as experimental_linear


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('reference');p.add_argument('--output',required=True)
    p.add_argument('--modes',nargs='+',default=['original','ieee','separate','fp64_tiles','tf32x3'],
        choices=['original','ieee','expert_ieee','attention_ieee','separate','fp64_tiles','tf32x3'])
    p.add_argument('--positions',nargs='+',type=int,default=[788]);p.add_argument('--fused',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest')
    reference=torch.load(args.reference,map_location='cpu',weights_only=False)
    if any(pos not in reference['positions'] for pos in args.positions):raise ValueError('missing reference position')
    context=1<<max(args.positions).bit_length()
    from scripts.experiments.gpt_oss_fused_attention import FusedAttentionExecutor
    model=ModelWeights(args.model,context);executor=(FusedAttentionExecutor if args.fused else GptOssExecutor)(model,use_graph=False,max_batch=1)
    original=executor.dense_linear;rows=[]
    permanent={weight.data_ptr() for weight in executor.w.values() if torch.is_tensor(weight)}
    attention={executor.w[f'{i}.{kind}'].data_ptr() for i in range(executor.c['num_hidden_layers']) for kind in ('qkv.weight','out')}
    for mode in args.modes:
        def project(x,w,bias=None):
            if mode=='original' or x.shape[0]<3 or w.dtype!=torch.float16:return original(x,w,bias)
            if mode=='expert_ieee' and w.data_ptr() in permanent:return original(x,w,bias)
            if mode=='attention_ieee' and w.data_ptr() not in attention:return original(x,w,bias)
            if mode in ('ieee','expert_ieee','attention_ieee'):return F.linear(x.float(),w.float(),None if bias is None else bias.float())
            result=experimental_linear(x,w,mode)
            return result if bias is None else result+bias
        executor.dense_linear=project
        for position in args.positions:
            ids=reference['ids'][:position+1];expected=reference['logits'][reference['positions'].index(position)]
            # First execution includes compilation; only the second is timed.
            executor.prefill(ids);torch.cuda.synchronize();start=time.perf_counter()
            actual=executor.prefill(ids)[0];torch.cuda.synchronize();seconds=time.perf_counter()-start
            actual=actual.cpu();error=actual-expected
            row={'mode':mode,'prefix_tokens':len(ids),'warm_prefill_seconds':seconds,
                'relative_rms':float(error.square().mean().sqrt()/expected.square().mean().sqrt()),
                'max_abs':float(error.abs().max()),'greedy_match':int(actual.argmax())==int(expected.argmax())}
            rows.append(row);print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'context':context,
                'reference':args.reference,'fused_attention':args.fused,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
