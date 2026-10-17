"""Independent encoder/reference checks for vectors, padding and batch composition."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
from damendaemon.encoder import EncoderWeights,EncoderExecutor
from damendaemon.runtime import require_gpu
from damendaemon.device import device_name


def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=p.parse_args()
    os.environ['DAMENDAEMON_DEVICE']=args.device
    require_gpu()
    torch.set_num_threads(4)
    model=EncoderWeights(args.path,2048)
    executor=EncoderExecutor(model)
    reference=AutoModel.from_pretrained(args.path,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').eval()
    tokenizer=AutoTokenizer.from_pretrained(args.path,local_files_only=True,trust_remote_code=False)
    # Read the reference preprocessing policy independently of native config.
    sentence_config=json.loads((Path(args.path)/'sentence_bert_config.json').read_text())
    lowercase=sentence_config.get('do_lower_case',False)
    settings_path=Path(args.path)/'config_sentence_transformers.json'
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
    truncate_dim=settings.get('truncate_dim')
    modules=json.loads((Path(args.path)/'modules.json').read_text())
    normalize=any(module['type'].endswith('.Normalize') for module in modules)
    pooling_config=json.loads((Path(args.path)/modules[1]['path']/'config.json').read_text())
    legacy_modes={'pooling_mode_cls_token':'cls','pooling_mode_max_tokens':'max',
                  'pooling_mode_mean_tokens':'mean','pooling_mode_mean_sqrt_len_tokens':'mean_sqrt_len_tokens',
                  'pooling_mode_weightedmean_tokens':'weightedmean','pooling_mode_lasttoken':'lasttoken'}
    pooling=pooling_config.get('pooling_mode') or [mode for flag,mode in legacy_modes.items() if pooling_config.get(flag)]
    default_name=settings.get('default_prompt_name')
    prefix=(settings.get('prompts',{}).get(default_name) or '') if default_name is not None else ''
    texts=['A dog is playing in the park.','A puppy chases a ball.','Financial markets fell today.',
           'こんにちは — café 🦙',' ','A long sentence '+('about a blue sky. '*20)]
    checks=[]
    def metrics(actual,expected):
        actual,expected=actual.float().cpu(),expected.float().cpu()
        assert actual.shape==expected.shape
        delta=actual-expected
        relative=(delta.square().mean(-1)/expected.square().mean(-1)).sqrt()
        cosine=F.cosine_similarity(actual,expected)
        absolute=delta.abs().amax(-1)
        assert torch.isfinite(actual).all()
        if normalize and (truncate_dim is None or truncate_dim>=model.config['hidden_size']*(1 if isinstance(pooling,str) else len(pooling))):
            assert torch.allclose(actual.norm(dim=-1),torch.ones(actual.shape[0]),atol=1e-5)
        assert (relative<.002).all() and (absolute<.001).all() and (cosine>.99999).all(),(relative,absolute,cosine)
        return {'relative_rms':relative.tolist(),'max_absolute':absolute.tolist(),'cosine_similarity':cosine.tolist()}
    @torch.inference_mode()
    def check(inputs):
        tokens=model.encode(inputs)
        for value,ids in zip(inputs,tokens):
            if isinstance(value,str):
                value=(prefix+value).strip()
                if lowercase:value=value.lower()
                assert ids==tokenizer.encode(value,add_special_tokens=True)
        count,length=len(tokens),max(map(len,tokens))
        ids=torch.full((count,length),model.config['pad_token_id'],dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,row in enumerate(tokens):ids[i,:len(row)]=torch.tensor(row);mask[i,:len(row)]=1
        hidden=reference(input_ids=ids,attention_mask=mask).last_hidden_state
        vectors=[]
        for row,sequence in enumerate(tokens):
            valid=hidden[row,:len(sequence)]
            parts=[]
            for mode in ([pooling] if isinstance(pooling,str) else pooling):
                if mode=='cls':vector=valid[0]
                elif mode=='lasttoken':vector=valid[-1]
                elif mode=='max':vector=valid.amax(0)
                elif mode=='mean_sqrt_len_tokens':vector=valid.sum(0)/(len(sequence)**.5)
                elif mode=='weightedmean':
                    weights=torch.arange(1,len(sequence)+1,dtype=torch.float32)
                    vector=(valid*weights[:,None]).sum(0)/weights.sum()
                else:vector=valid.mean(0)
                parts.append(vector)
            vector=torch.cat(parts)
            vectors.append(vector)
        expected=torch.stack(vectors)
        if normalize:expected=F.normalize(expected,p=2,dim=-1)
        if truncate_dim is not None:expected=expected[:,:truncate_dim]
        actual=executor.embed(tokens).cpu()
        checks.append({'lengths':list(map(len,tokens)),**metrics(actual,expected)})
        return actual
    singles=torch.cat([check([text]) for text in texts])
    mixed=check(texts)
    checks.append({'kind':'batch_invariance',**metrics(mixed,singles)})
    check([texts[5],texts[0],texts[4]])
    seed=tokenizer.encode('blue',add_special_tokens=False)[0]
    check([[tokenizer.cls_token_id]+[seed]*(length-2)+[tokenizer.sep_token_id] for length in (2,127,255,256)])
    assert torch.dot(mixed[0],mixed[1])>torch.dot(mixed[0],mixed[2])
    # Replay with different input data and churn the bounded cache. Returned
    # device vectors must remain valid even when a subsequent replay completes.
    if executor.use_graph:
        first_tokens=model.encode([texts[0]])
        retained=executor.embed(first_tokens)
        snapshot=retained.clone()
        executor.embed([[tokenizer.cls_token_id]+[seed]*(len(first_tokens[0])-2)+[tokenizer.sep_token_id]])
        assert torch.equal(retained,snapshot), 'graph replay mutated returned vectors'
        for length in (3,7,15,31,63,127):
            check([[tokenizer.cls_token_id]+[seed]*(length-2)+[tokenizer.sep_token_id]])
            assert len(executor.graphs)<=4
        check([texts[0]])
    for bad in ([[seed]*257],[[model.config['vocab_size']]],[[True]],[[1.5]],[]):
        try:model.encode(bad);raise AssertionError('invalid encoder input accepted')
        except ValueError:pass
    print(json.dumps({'passed':True,'model_path':args.path,'device':device_name(),'compute_dtype':str(executor.compute_dtype),
        'reference_dtype':'float32','reference_attention':'eager','host_weight_bytes':model.bytes,
        'embedding_dimensions':model.config['embedding_dimension'],'saved_truncate_dim':truncate_dim,
        'normalize_module':normalize,'pooling_mode':pooling,'max_input_tokens':model.max_context,
        'semantic_ordering':True,'checks':checks},indent=2))


if __name__=='__main__':main()
