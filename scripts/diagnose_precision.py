"""Compare suspect FP16 drift with an independent FP32 reference; not an acceptance gate."""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM
from damendaemon.qwen2 import ModelWeights, Executor
from damendaemon.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    require_gpu()
    model = ModelWeights(args.path, 256)
    executor = Executor(model, use_graph=True)
    refs = [AutoModelForCausalLM.from_pretrained(args.path, local_files_only=True, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation='sdpa').cuda().eval() for dtype in (torch.float16, torch.float32)]
    ids = model.chat_tokens([{'role': 'user', 'content': 'Reply in French: Hello, how are you?'}])
    rows = []
    def error(a, b):
        a, b = a.float(), b.float()
        delta = a-b
        return {'relative_rms': (delta.square().mean().sqrt()/b.square().mean().sqrt()).item(),
                'offset': delta.mean().item(),
                'centered_relative_rms': ((delta-delta.mean()).square().mean().sqrt()/(b-b.mean()).square().mean().sqrt()).item(),
                'max_absolute': delta.abs().max().item(), 'greedy_match': a.argmax().item()==b.argmax().item()}
    with torch.inference_mode():
        results = [ref(torch.tensor([ids], device='cuda'), use_cache=True, logits_to_keep=1) for ref in refs]
        actual = executor.prefill(ids)
        for step in range(8):
            targets = [r.logits[:,-1] for r in results]
            rows.append({'step':step, 'damendaemon_vs_fp16':error(actual,targets[0]),
                         'damendaemon_vs_fp32':error(actual,targets[1]), 'reference_fp16_vs_fp32':error(targets[0],targets[1])})
            token = targets[1].argmax().item()
            results = [ref(torch.tensor([[token]],device='cuda'),past_key_values=r.past_key_values,
                           use_cache=True,logits_to_keep=1) for ref,r in zip(refs,results)]
            actual,_=executor.decode(token)
    print(json.dumps({'model':args.path,'diagnostic_only':True,'rows':rows},indent=2))


if __name__ == '__main__':
    main()
