"""Check packed native GPT-OSS against independent streamed CPU reference layers."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from dllm.qwen2 import ModelWeights
from dllm.gpt_oss import GptOssExecutor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('reference')
    parser.add_argument('--output',required=True);parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--scalar',action='store_true',help='exercise direct single-request decode in slot zero')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    reference=torch.load(args.reference,map_location='cpu',weights_only=False)
    model=ModelWeights(args.model,256);executor=GptOssExecutor(model,use_graph=not args.no_graph,max_batch=2)
    checks=[];timings=[]
    def compare(actual,expected,kind,**metadata):
        actual=actual.detach().cpu();expected=expected.float()
        error=actual-expected
        rms=float(error.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-9))
        maximum=float(error.abs().max())
        greedy=bool(torch.equal(actual.argmax(-1),expected.argmax(-1))) if kind!='layer' else None
        passed=bool(torch.isfinite(actual).all()) and rms<.01 and maximum<.5 and greedy is not False
        checks.append({'kind':kind,**metadata,'relative_rms':rms,'max_absolute_error':maximum,
            'greedy_match':greedy,'passed':passed})
        if not passed:print(json.dumps(checks[-1]),flush=True)
    with torch.inference_mode():
        for index,case in enumerate(reference['cases']):
            ids=case['ids'];count=len(ids);split=case['split'];slot=0 if args.scalar else index%2
            positions=torch.arange(count,device='cuda');slots=torch.full_like(positions,slot)
            x=torch.nn.functional.embedding(torch.tensor(ids,device='cuda'),executor.w['embed']).float()
            start=time.perf_counter()
            for layer in range(model.config['num_hidden_layers']):
                x=executor.layer(x,positions,slots,layer,prefill=True)
                compare(x,case['layers'][layer],'layer',case=index,layer=layer)
            logits=executor.dense_linear(executor.norm(x[-1:],executor.w['norm']),executor.w['head'])
            compare(logits[0],case['logits'][-1],'full_prefill',case=index)
            timings.append({'case':index,'kind':'all_layers_with_checks','seconds':time.perf_counter()-start})
            print(json.dumps({'phase':'layers_checked','case':index,'seconds':timings[-1]['seconds']}),flush=True)
            other=1-slot;executor.prefill(ids[:3],other)
            untouched=[cache[:,other].clone() for cache in executor.kv]
            start=time.perf_counter();logits=executor.prefill(ids[:split],slot);torch.cuda.synchronize()
            timings.append({'case':index,'kind':'prefill','tokens':split,'seconds':time.perf_counter()-start})
            compare(logits[0],case['logits'][0],'prefix',case=index)
            for step,token in enumerate(ids[split:]):
                start=time.perf_counter();logits,_=executor.decode(token) if args.scalar else executor.decode_batch([token],[slot]);torch.cuda.synchronize()
                timings.append({'case':index,'kind':'decode','step':step,'seconds':time.perf_counter()-start})
                compare(logits[0],case['logits'][step+1],'decode',case=index,step=step)
            isolated=all(torch.equal(cache[:,other],saved) for cache,saved in zip(executor.kv,untouched))
            checks.append({'kind':'other_slot_unchanged','case':index,'passed':isolated})
            del untouched
        # Both slots have prior content. Reusing either for a short prompt must
        # not expose stale entries left by the longer prompt.
        case=reference['cases'][0]
        for slot in (0,1):
            actual=executor.prefill(case['ids'][:case['split']],slot)
            compare(actual[0],case['logits'][0],'slot_reuse',slot=slot)
    report={'passed':all(row['passed'] for row in checks),'gpu':torch.cuda.get_device_name(),
        'repository':reference['repository'],'revision':reference['revision'],
        'reference':reference['reference'],'reference_transformers':reference['transformers'],
        'reference_torch':reference['torch'],'reference_dtype':reference['dtype'],'torch':str(torch.__version__),
        'graph':not args.no_graph,'scalar':args.scalar,'prefill_mode':executor.grouped_prefill,'chunked_experts':executor.chunked_experts,'compute_dtype':str(executor.compute_dtype),'kernel_backend':executor.kernel_backend,
        'context':executor.context,'max_batch':executor.max_batch,'host_weight_bytes':model.bytes,
        'allocated_bytes':torch.cuda.memory_allocated(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'checks':checks,'timings':timings}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'checks':len(checks),'output':args.output}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
