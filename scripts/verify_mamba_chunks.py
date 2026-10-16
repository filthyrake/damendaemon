"""Real GPU chunk/prefix proof against full CPU reference recomputation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import torch
from reference_model import load_reference
from dllm.models import load
from dllm.runtime import require_gpu,make_executor
from dllm.prefix import PrefixCache


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--quantization',choices=('none','q8_0'),default='none')
    p.add_argument('--chunk-size',type=int,default=127)
    args=p.parse_args();os.environ['DLLM_DEVICE']=args.device
    if not 1<=args.chunk_size<=2048:p.error('chunk size must be 1..2048')
    require_gpu();torch.set_num_threads(4)
    model=load(args.model,2048,quantization=args.quantization)
    executor=make_executor(model,max_batch=4)
    reference=load_reference(args.model,reference_quantization=args.quantization,
        local_files_only=True,torch_dtype=torch.float32,
        **({'chunk_size':16} if model.config['model_type']=='mamba2' else {})).eval()
    corpus=model.raw_tokens('The blue bird flew over the green field while a dog slept under a tree. '*200)
    checks=[]
    def check(ids,slot,logits,label):
        ref=reference(torch.tensor([ids]),use_cache=True)
        expected=ref.logits[0,-1];actual=logits[0].cpu()
        relative=((actual-expected).square().mean()/expected.square().mean()).sqrt().item()
        assert relative<.001 and (actual-expected).abs().max()<.03,(label,relative)
        assert actual.argmax()==expected.argmax(),label
        for layer in (0,model.config['num_hidden_layers']//2,model.config['num_hidden_layers']-1):
            target=ref.cache_params.ssm_states[layer][0].float().reshape_as(executor.state[layer][slot])
            value=executor.state[layer][slot].cpu()
            ratio=((value-target).square().mean()/target.square().mean().clamp_min(1.e-12)).sqrt()
            assert ratio<.001,(label,layer,ratio.item())
            torch.testing.assert_close(executor.conv[layer][slot].cpu(),ref.cache_params.conv_states[layer][0],atol=.001,rtol=.001)
        checks.append({'label':label,'length':len(ids),'relative_rms':relative,'greedy_match':True})

    prompts={3:corpus[:257],1:corpus[5:86]};offsets={3:0,1:0}
    # Interleave slots, cross the convolution width, and leave intermediate
    # chunks without output projection. Each boundary reference starts fresh.
    for slot,size,final in ((3,1,True),(1,3,True),(3,2,True),(1,7,False),
                            (3,17,False),(1,71,True),(3,237,True)):
        start=offsets[slot];ids=prompts[slot][start:start+size]
        logits=executor.prefill_chunk(ids,slot,start,final=final)
        offsets[slot]+=len(ids)
        if final:check(prompts[slot][:offsets[slot]],slot,logits,'interleaved-chunk')
        else:assert logits is None
    prompt=prompts[3];cache=PrefixCache(executor,600,max_entries=2)
    cache.remember(prompt,3,logits)
    count,saved=cache.restore(prompt,0)
    assert count==len(prompt);check(prompt,0,saved,'exact-prefix')
    saved.zero_()
    token=logits.argmax(-1).item()
    decoded,_=executor.decode_batch([token],[0])
    check(prompt+[token],0,decoded,'decode-after-prefix')
    count,saved=cache.restore(prompt,1)
    assert count==len(prompt);check(prompt,1,saved,'snapshot-survives-other-slot-decode')
    suffix=corpus[300:317];count,saved=cache.restore(prompt+suffix,2)
    assert count==len(prompt) and saved is None
    extended=executor.prefill_chunk(suffix,2,count)
    check(prompt+suffix,2,extended,'extend-prefix')
    assert cache.restore(prompt[:-1],0)==(0,None)
    assert cache.restore(prompt[:-1]+[0],0)==(0,None)
    before=list(executor.lengths)
    for start,tokens in ((len(prompt)-1,[1]),(2048,[1]),(len(prompt),[])):
        try:executor.prefill_chunk(tokens,1,start)
        except ValueError:pass
        else:raise AssertionError('invalid continuation accepted')
    assert before==executor.lengths
    long=corpus[:2048]
    for start in range(0,len(long),args.chunk_size):
        logits=executor.prefill_chunk(long[start:start+args.chunk_size],0,start,final=start+args.chunk_size>=len(long))
    check(long,0,logits,'full-context-chunked')
    cache.remember(long,0,logits)
    assert cache.tokens==len(prompt),'oversized recurrent snapshot was truncated'
    for slot in (0,2):
        fresh=corpus[20+slot:23+slot]
        logits=executor.prefill_chunk(fresh,slot,0)
        check(fresh,slot,logits,'reset-after-reuse')
    print(json.dumps({'passed':True,'device':args.device,'model':args.model,
        'quantization':args.quantization,'checks':checks,'state_checks':True,
        'prefix_no_rewind':True,'independent_snapshot':True,'invalid_state_rejected':True,
        'snapshot_bytes':executor.prefix_state_bytes(len(prompt)),
        'architecture':model.config['model_type'],
        'full_context_chunk_size':args.chunk_size,
        'prefill_ssd':getattr(executor,'prefill_ssd',False),
        'reference_chunk_size':16 if model.config['model_type']=='mamba2' else None,
        'reference':'CPU full prompt recomputation; independently quantized matrices when selected'},indent=2))


if __name__=='__main__':main()
