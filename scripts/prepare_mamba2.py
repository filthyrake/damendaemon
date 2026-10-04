"""Convert the published Mamba-2 single-group checkpoint for independent testing.

Configuration mapping follows Hugging Face's Apache-2.0 Mamba2 converter:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/convert_mamba2_ssm_checkpoint_to_pytorch.py
This offline utility uses Transformers to validate/save weights, never to serve.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import Mamba2Config,Mamba2ForCausalLM,AutoTokenizer
from safetensors.torch import save_model


def main():
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('output')
    p.add_argument('--tokenizer',required=True);args=p.parse_args()
    source=Path(args.original);c=json.loads((source/'config.json').read_text())
    if (c['ssm_cfg']!={'layer':'Mamba2'} or c.get('attn_layer_idx') or c.get('attn_cfg')
            or c.get('d_intermediate',0)!=0 or c.get('rms_norm') is not True
            or c.get('residual_in_fp32') is not True):
        raise ValueError('only original pure single-group Mamba-2 is qualified')
    multiple=c['pad_vocab_size_multiple'];vocab=((c['vocab_size']+multiple-1)//multiple)*multiple
    config=Mamba2Config(hidden_size=c['d_model'],num_hidden_layers=c['n_layer'],
        num_heads=c['d_model']*2//64,head_dim=64,n_groups=1,state_size=128,
        vocab_size=vocab,tie_word_embeddings=c['tie_embeddings'],bos_token_id=0,eos_token_id=0,pad_token_id=0)
    torch.set_num_threads(4)
    model=Mamba2ForCausalLM(config)
    weights=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True)
    weights={name.replace('backbone.embedding.','backbone.embeddings.'):value for name,value in weights.items()}
    model.load_state_dict(weights,strict=True)
    for name,value in model.state_dict().items():
        if not torch.equal(value,weights[name].float()):
            raise ValueError('conversion changed original tensor: '+name)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config.architectures=['Mamba2ForCausalLM'];config.torch_dtype=torch.float32
    config.save_pretrained(output)
    save_model(model.float(),str(output/'model.safetensors'),metadata={'format':'pt'})
    AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True).save_pretrained(args.output)
    with (source/'pytorch_model.bin').open('rb') as file:
        source_hash=hashlib.file_digest(file,'sha256').hexdigest()
    print(json.dumps({'source':str(source),'output':args.output,'source_dtype':str(next(iter(weights.values())).dtype),
        'saved_dtype':'float32','strict_state_dict':True,'all_original_tensors_exact':True,
        'source_sha256':source_hash,
        'config':config.to_dict()},indent=2))


if __name__=='__main__':main()
