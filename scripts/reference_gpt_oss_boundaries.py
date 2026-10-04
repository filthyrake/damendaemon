"""Independent CPU output logits at specifically selected routing-sensitive positions."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('trace')
    parser.add_argument('--positions',nargs='+',type=int,required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);root=Path(args.model)
    reference=torch.load(args.trace,map_location='cpu',weights_only=False);case=reference['cases'][0]
    if any(p<0 or p>=len(case['ids']) for p in args.positions):raise ValueError('position outside reference')
    source=json.loads((root/'dllm-source.json').read_text());catalog=source['tensors'];config=source['config']
    def get(key):
        with safe_open(str(root/catalog[key]['file']),framework='pt',device='cpu') as file:return file.get_tensor(key).float()
    norm=GptOssRMSNorm(config['hidden_size'],eps=config['rms_norm_eps']);norm.load_state_dict({'weight':get('model.norm.weight')})
    logits=torch.nn.functional.linear(norm(case['layers'][-1][args.positions]),get('lm_head.weight'))
    torch.save({'repository':reference['repository'],'revision':reference['revision'],'source_reference':str(args.trace),
        'positions':args.positions,'ids':case['ids'],'logits':logits,'source_device':reference['device'],
        'reference':'Unmodified Transformers RMS norm and CPU FP32 output head over independent saved final-layer states'},args.output)


if __name__=='__main__':main()
