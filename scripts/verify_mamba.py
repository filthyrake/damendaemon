"""Independent FP32 Mamba logits/recurrent state and reused-slot proof."""
import argparse
import json
import os
import torch
from reference_model import load_reference
from dllm.models import load
from dllm.runtime import require_gpu,make_executor


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda');p.add_argument('--context',type=int,default=512)
    p.add_argument('--reference-quantization',choices=('none','q8_0'),default='none')
    args=p.parse_args();os.environ['DLLM_DEVICE']=args.device
    require_gpu();torch.set_num_threads(4)
    model=load(args.model,args.context);executor=make_executor(model,max_batch=4)
    assert model.quantization==args.reference_quantization
    reference=load_reference(args.model,reference_quantization=args.reference_quantization,local_files_only=True,torch_dtype=torch.float32,
        **({'chunk_size':16} if model.config['model_type']=='mamba2' else {})).eval()
    corpus=model.raw_tokens('The blue bird flew over the green field while a dog slept under a tree. '*200)
    states={};checks=[]
    def compare(actual,target,slot,phase):
        actual=actual.float().cpu();target=target.float().cpu();error=actual-target
        relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1.e-6)).item()
        absolute=error.abs().max().item();match=actual.argmax().item()==target.argmax().item()
        checks.append({'slot':slot,'phase':phase,'relative_rms':relative,'max_absolute':absolute,
            'greedy_match':match,'passed':relative<.001 and absolute<.03 and match})
    def state_check(slot,cache):
        for layer in (0,model.config['num_hidden_layers']//2,model.config['num_hidden_layers']-1):
            actual=executor.state[layer][slot].cpu();expected=cache.ssm_states[layer][0].float().reshape_as(actual)
            relative=((actual-expected).square().mean()/expected.square().mean().clamp_min(1.e-12)).sqrt().item()
            assert relative<.001,(slot,layer,relative)
            torch.testing.assert_close(executor.conv[layer][slot].cpu(),cache.conv_states[layer][0],atol=.001,rtol=.001)
    def prefill(lengths,slots):
        prompts=[corpus[row*3:row*3+length] for row,length in enumerate(lengths)]
        actual=executor.prefill_batch(prompts,slots)
        for row,(slot,prompt) in enumerate(zip(slots,prompts)):
            result=reference(torch.tensor([prompt]),use_cache=True)
            compare(actual[row],result.logits[0,-1],slot,'prefill:'+str(len(prompt)))
            state_check(slot,result.cache_params)
            states[slot]=(result.cache_params,len(prompt),result.logits[0,-1].argmax().item())
    def decode(slots,count):
        for _ in range(count):
            actual,_=executor.decode_batch([states[s][2] for s in slots],slots)
            for row,slot in enumerate(slots):
                cache,length,token=states[slot]
                # HF's cache_position indexes the convolution ring, not prompt
                # positions. generate() advances from conv_kernel after prefill,
                # including when the prompt is shorter than that kernel.
                position=max(length,model.config['conv_kernel'])
                result=reference(torch.tensor([[token]]),cache_params=cache,cache_position=torch.tensor([position]),use_cache=True)
                compare(actual[row],result.logits[0,-1],slot,'decode:'+str(length))
                states[slot]=(result.cache_params,length+1,result.logits[0,-1].argmax().item())
    prefill([1,3,4,17],[3,0,2,1]);decode([2,3,1,0],8)
    prefill([31,17,31,17],[1,3,0,2]);decode([3,0,2,1],8)
    prefill([args.context-8,31],[0,1]);decode([1,2,0,3],8)
    for slot in states:state_check(slot,states[slot][0])
    before=list(executor.lengths)
    try:executor.decode_batch([states[0][2]],[0])
    except ValueError:pass
    else:raise AssertionError('exhausted recurrent context accepted')
    assert executor.lengths==before
    prefill([3],[0]);decode([0],32)
    prompt=model.raw_tokens('The capital of France is')
    logits=executor.prefill(prompt);generated=[]
    for step in range(16):
        token=logits.argmax(-1).item();generated.append(token)
        if token in model.eos:break
        if step<15:logits,_=executor.decode(token)
    expected=reference.generate(torch.tensor([prompt]),attention_mask=torch.ones((1,len(prompt)),dtype=torch.long),max_new_tokens=16,do_sample=False)[0].tolist()[len(prompt):]
    assert generated==expected,(generated,expected)
    passed=all(c['passed'] for c in checks)
    print(json.dumps({'passed':passed,'model':args.model,'device':args.device,'context':args.context,
        'kernel_backend':executor.kernel_backend,'fused_time_step':executor.fuse_time_step,
        'checks':checks,'recurrent_state_checks':True,
        'slot_reuse':True,'generated_ids':generated,'generated_text':model.tokenizer.decode(generated),
        'reference_quantization':args.reference_quantization,
        'architecture':model.config['model_type'],
        'prefill_ssd':getattr(executor,'prefill_ssd',False),
        'reference_chunk_size':16 if model.config['model_type']=='mamba2' else None,
        'reference':'CPU FP32 Transformers; independent GGUF quantization when selected'},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
