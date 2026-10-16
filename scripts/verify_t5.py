"""Independent encoder-decoder logits, cache isolation and slot-reuse proof."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import torch
from transformers import AutoModelForSeq2SeqLM
from dllm.models import load
from dllm.runtime import require_gpu,make_executor


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--full-decoder',action='store_true')
    args=p.parse_args();os.environ['DLLM_DEVICE']=args.device
    require_gpu();torch.set_num_threads(4)
    model=load(args.model,args.context)
    executor=make_executor(model,max_batch=4)
    reference=AutoModelForSeq2SeqLM.from_pretrained(args.model,local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float32).eval()
    corpus=model.raw_tokens('summarize: The blue bird flies over the green field. '*100)
    states={};checks=[]
    def compare(actual,target,kind,slot):
        actual=actual.float().cpu();target=target.float().cpu()
        error=actual-target
        relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1.e-6)).item()
        absolute=error.abs().max().item()
        match=actual.argmax().item()==target.argmax().item()
        checks.append({'kind':kind,'slot':slot,'relative_rms':relative,'max_absolute':absolute,
            'greedy_match':match,'passed':relative<.001 and absolute<.02 and match})
    def prefill(lengths,slots):
        prompts=[(corpus*4)[:length] for length in lengths]
        actual=executor.prefill_batch(prompts,slots)
        for row,(slot,prompt) in enumerate(zip(slots,prompts)):
            result=reference(input_ids=torch.tensor([prompt]),
                decoder_input_ids=torch.tensor([[model.config['decoder_start_token_id']]]),use_cache=True)
            compare(actual[row],result.logits[0,-1],'prefill:'+str(len(prompt)),slot)
            states[slot]=(result.encoder_last_hidden_state,result.past_key_values,result.logits[0,-1].argmax().item())
    def decode(slots,count):
        for _ in range(count):
            tokens=[states[slot][2] for slot in slots]
            actual,_=executor.decode_batch(tokens,slots)
            for row,(slot,token) in enumerate(zip(slots,tokens)):
                encoded,cache,_=states[slot]
                result=reference(encoder_outputs=(encoded,),past_key_values=cache,
                    decoder_input_ids=torch.tensor([[token]]),use_cache=True)
                compare(actual[row],result.logits[0,-1],'decode:'+str(executor.lengths[slot]),slot)
                states[slot]=(encoded,result.past_key_values,result.logits[0,-1].argmax().item())
    prefill([1,15,17,127],[3,0,2,1]);decode([2,3,1,0],8)
    prefill([args.context,31],[0,1]);decode([1,2,0,3],8)
    decode([0],128)
    if args.full_decoder:
        decode([0],args.context-executor.lengths[0])
        try:executor.decode_batch([states[0][2]],[0])
        except ValueError:pass
        else:raise AssertionError('exhausted decoder cache accepted')
    before=list(executor.lengths)
    try:executor.prefill_batch([[1]*(args.context+1)],[0])
    except ValueError:pass
    else:raise AssertionError('oversized encoder input accepted')
    assert executor.lengths==before
    prompt=model.raw_tokens('translate English to German: How old are you?')
    logits=executor.prefill(prompt);generated=[]
    for _ in range(32):
        token=logits.argmax(-1).item();generated.append(token)
        if token in model.eos:break
        logits,_=executor.decode(token)
    expected=reference.generate(torch.tensor([prompt]),max_new_tokens=32,do_sample=False)[0].tolist()[1:]
    assert generated==expected,(generated,expected)
    passed=all(check['passed'] for check in checks)
    print(json.dumps({'passed':passed,'model':args.model,'device':args.device,'context':args.context,
        'kernel_backend':executor.kernel_backend,'compute_dtype':str(executor.compute_dtype),
        'checks':checks,'encoder_decoder_lengths_independent':True,'slot_reuse':True,
        'full_decoder_boundary':args.full_decoder,
        'generated_ids':generated,'generated_text':model.tokenizer.decode(generated,skip_special_tokens=True),
        'reference_dtype':'float32','reference':'Transformers T5ForConditionalGeneration on CPU'},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
