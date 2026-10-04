"""Check real tokenizer grammar masks, including padded vocabulary and Unicode."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from tokenizers import Tokenizer
from dllm.config import decoder_config,eos_ids
from dllm.structured import Constraint,response_schema


def main():
    p=argparse.ArgumentParser()
    p.add_argument('path',type=Path)
    p.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    args=p.parse_args()
    config=decoder_config(json.loads((args.path/'config.json').read_text()))
    model=SimpleNamespace(config=config,tokenizer=Tokenizer.from_file(str(args.path/'tokenizer.json')),
                          eos=eos_ids(config['eos_token_id']))
    # Continuations do not run the tokenizer's input-only dummy-prefix insertion.
    tokenizer_data=json.loads(model.tokenizer.to_str())
    normalizer=tokenizer_data.get('normalizer')
    if normalizer and normalizer.get('type')=='Sequence':
        normalizer['normalizers']=[n for n in normalizer['normalizers'] if n.get('type')!='Prepend']
    continuation_tokenizer=Tokenizer.from_str(json.dumps(tokenizer_data))
    checks=[]
    for value in ({'text':'café 🦙\n"\\','flag':True}, {'nested':[None,1,-2,3.25,{'key':'value'}]}):
        constraint=Constraint(model,response_schema({'type':'json_object'}))
        encoded=json.dumps(value,ensure_ascii=False,separators=(',',':'))
        tokens=continuation_tokenizer.encode(encoded,add_special_tokens=False).ids
        for token in tokens:
            logits=torch.zeros(config['vocab_size'],device=args.device)
            masked=constraint.apply(logits)
            assert torch.isfinite(masked[token])
            assert torch.isfinite(logits).all()
            # Compare every vocabulary bit with the independent CPU mask application.
            reference=logits.cpu().clone()
            constraint.xgr.apply_token_bitmask_inplace(reference.unsqueeze(0),constraint.mask,backend='cpu')
            assert torch.equal(torch.isfinite(masked).cpu(),torch.isfinite(reference))
            constraint.accept(token)
        stop=next(iter(model.eos))
        assert torch.isfinite(constraint.apply(torch.zeros(config['vocab_size'],device=args.device))[stop])
        constraint.accept(stop)
        assert constraint.matcher.is_terminated()
        checks.append({'text':encoded,'tokens':len(tokens),'passed':True})
    print(json.dumps({'passed':True,'model_path':str(args.path),'device':args.device,'checks':checks},indent=2))


if __name__=='__main__':main()
