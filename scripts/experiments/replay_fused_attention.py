"""Apply unchanged numerical replay gates to the isolated fused-attention executor."""
from functools import partial
import json
from pathlib import Path
import sys
import torch
from scripts import replay_gpt_oss
from scripts.experiments.gpt_oss_fused_attention import FusedAttentionExecutor


def main():
    ieee='--ieee-attention' in sys.argv
    if ieee:sys.argv.remove('--ieee-attention')
    torch.set_float32_matmul_precision('highest')
    original=replay_gpt_oss.GptOssExecutor
    replay_gpt_oss.GptOssExecutor=partial(FusedAttentionExecutor,ieee_attention=ieee)
    try:replay_gpt_oss.main()
    finally:
        replay_gpt_oss.GptOssExecutor=original
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])
            if path.exists():
                result=json.loads(path.read_text());result['experimental_attention']='bounded_fused_sink'
                result['experimental_ieee_attention']=ieee
                path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
