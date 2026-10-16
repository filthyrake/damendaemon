"""Compare static rotary schedules with the independent Transformers reference."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import torch
import transformers
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from dllm.rotary import inverse_frequencies, attention_factor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    args=parser.parse_args()
    results=[]
    for dim in (64,96,128):
        for variant in ('default','linear','llama3','yarn','yarn-original','yarn-explicit','yarn-equal-bounds'):
            kind='yarn' if variant.startswith('yarn') else variant
            scaling=None if kind=='default' else {'rope_type':kind,'factor':32.0}
            if kind=='llama3':
                scaling.update(low_freq_factor=1.0,high_freq_factor=4.0,original_max_position_embeddings=8192)
            if variant=='yarn-original':
                scaling.update(original_max_position_embeddings=8192)
            if variant=='yarn-explicit':
                scaling.update(attention_factor=1.3,beta_fast=16.,beta_slow=2.)
            if variant=='yarn-equal-bounds':
                scaling.update(beta_fast=4.,beta_slow=4.)
            config={'head_dim':dim,'rope_theta':500000.0,'rope_scaling':scaling,'max_position_embeddings':131072}
            actual=inverse_frequencies(config,args.device)
            reference=LlamaConfig(hidden_size=dim*8,num_attention_heads=8,**config)
            expected,expected_scale=ROPE_INIT_FUNCTIONS[kind](reference,torch.device(args.device))
            torch.testing.assert_close(actual,expected,atol=1.e-9,rtol=1.e-6)
            scale=attention_factor(config)
            assert abs(scale-expected_scale)<1.e-12
            positions=torch.tensor([0,1,2047,8191,8192,32768,131071],device=args.device,dtype=torch.float32)
            for operation in (torch.cos,torch.sin):
                torch.testing.assert_close(operation(torch.outer(positions,actual))*scale,
                    operation(torch.outer(positions,expected))*expected_scale,atol=1.e-6,rtol=1.e-6)
            results.append({'kind':kind,'variant':variant,'attention_factor':scale,'head_dim':dim,'max_position':131071})
    print(json.dumps({'passed':True,'device':args.device,'torch':str(torch.__version__),
                     'transformers':transformers.__version__,'checks':results},indent=2))


if __name__=='__main__':
    main()
