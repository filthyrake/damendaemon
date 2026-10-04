"""Check sensitive-prefix outputs without masking failed numerical gates."""
import argparse
import json
import os
from pathlib import Path
import torch
from dllm.qwen2 import ModelWeights
from dllm.gpt_oss import GptOssExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('reference');p.add_argument('--output',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    reference=torch.load(args.reference,map_location='cpu',weights_only=False)
    context=1<<max(reference['positions']).bit_length();model=ModelWeights(args.model,context)
    executor=GptOssExecutor(model,max_batch=1);rows=[]
    for index,position in enumerate(reference['positions']):
        ids=reference['ids'][:position+1]
        actual=executor.prefill(ids)[0].cpu();executor.prefill_attention_bytes=None
        unbounded=executor.prefill(ids)[0].cpu();executor.prefill_attention_bytes=32*1024**2
        exact=torch.equal(actual,unbounded);assert exact
        expected=reference['logits'][index];error=actual-expected
        relative=float(error.square().mean().sqrt()/expected.square().mean().sqrt());maximum=float(error.abs().max())
        row={'prefix_tokens':position+1,'relative_rms':relative,'max_absolute_error':maximum,
            'greedy_match':int(actual.argmax())==int(expected.argmax()),'bounded_unbounded_exact':exact,
            'native_token':int(actual.argmax()),'reference_token':int(expected.argmax()),
            'passed':relative<.01 and maximum<.5 and int(actual.argmax())==int(expected.argmax())}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'passed':all(row['passed'] for row in rows),'gpu':torch.cuda.get_device_name(),'context':context,'rows':rows,
        'reference':reference['reference'],'source_device':reference['source_device']}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
