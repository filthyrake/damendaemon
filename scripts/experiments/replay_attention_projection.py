"""Run unchanged GPT-OSS numerical gates with experimental FP32 attention products.

Only this verification process substitutes the executor constructor. Serving
continues to use its existing projection policy.
"""
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from scripts import replay_gpt_oss


def main():
    original=replay_gpt_oss.GptOssExecutor
    torch.set_float32_matmul_precision('highest')
    def create(*args,**kwargs):
        executor=original(*args,**kwargs);linear=executor.dense_linear
        weights={executor.w[f'{i}.{kind}'].data_ptr() for i in range(executor.c['num_hidden_layers']) for kind in ('qkv.weight','out')}
        def project(x,w,bias=None):
            if x.shape[0]>=3 and w.dtype==torch.float16 and w.data_ptr() in weights:
                return F.linear(x.float(),w.float(),None if bias is None else bias.float())
            return linear(x,w,bias)
        executor.dense_linear=project
        return executor
    replay_gpt_oss.GptOssExecutor=create
    try:replay_gpt_oss.main()
    finally:
        replay_gpt_oss.GptOssExecutor=original
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])
            if path.exists():
                result=json.loads(path.read_text());result['experimental_projection_policy']='attention_only_ieee_fp32'
                path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
