"""Independent routed-kernel and synthetic MoE architecture gates on CUDA."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
from pathlib import Path
import tempfile
import torch
from tokenizers import Tokenizer, models
from transformers import OlmoeConfig, OlmoeForCausalLM
from dllm.qwen2 import ModelWeights
from dllm.runtime import make_executor, require_gpu
from dllm.quantization import quantize, dequantize
from dllm.moe_kernels import routed_linear, combine


def main():
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(23)
    checks=[]
    for batch in (1,4):
        selected=torch.tensor([[2,0],[1,2],[0,1],[2,1]][:batch],device='cuda')
        probabilities=torch.softmax(torch.randn(batch,2,device='cuda'),-1)
        for quantized in (False,True):
            original=torch.randn(3*67,96).half()
            if quantized:
                weight,scales=quantize(original)
                weight,scales=weight.cuda(),scales.cuda()
                reference_weight=dequantize(weight,scales).view(3,67,96)
            else:
                weight,scales=original.cuda(),None
                reference_weight=weight.float().view(3,67,96)
            for shared in (False,True):
                x=torch.randn(batch if shared else batch*2,96,device='cuda')
                expected=torch.stack([torch.nn.functional.linear(x[row if shared else row*2+route],reference_weight[expert])
                    for row,experts in enumerate(selected.tolist()) for route,expert in enumerate(experts)])
                actual=routed_linear(x,weight,scales,selected,67,shared)
                torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
                expected_sum=(expected.view(batch,2,67)*probabilities[:,:,None]).sum(1)
                torch.testing.assert_close(combine(actual,probabilities),expected_sum,rtol=1e-4,atol=1e-4)
                checks.append(dict(kind='routed_kernel',batch=batch,quantized=quantized,shared_input=shared))
    for normalized,clip in ((False,None),(True,.75)):
        config=OlmoeConfig(vocab_size=256,hidden_size=64,intermediate_size=96,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=256,num_experts=4,num_experts_per_tok=2,
            norm_topk_prob=normalized,clip_qkv=clip,eos_token_id=1)
        config._attn_implementation='eager'
        reference=OlmoeForCausalLM(config).float().eval()
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({str(i):i for i in range(256)},unk_token='0')).save(str(Path(directory)/'tokenizer.json'))
            (Path(directory)/'tokenizer_config.json').write_text('{}')
            model=ModelWeights(directory,128)
            runtime=make_executor(model,max_batch=4)
            states={}
            def check(actual,target,kind,slot):
                error=actual.float().cpu()-target
                relative=(error.square().mean()/target.square().mean()).sqrt().item()
                match=actual.argmax().item()==target.argmax().item()
                assert relative<.01 and match and error.abs().max()<.5
                checks.append(dict(kind=kind,slot=slot,normalized=normalized,clip=clip,relative_rms=relative,greedy_match=match))
            with torch.inference_mode():
                prompts=[list(range(3,34)),list(range(5,70))]
                slots=[3,1]
                actual=runtime.prefill_batch(prompts,slots)
                for row,(ids,slot) in enumerate(zip(prompts,slots)):
                    result=reference(torch.tensor([ids]),use_cache=True)
                    states[slot]=result
                    check(actual[row],result.logits[:,-1],'prefill',slot)
                for active in ([1,3],[3],[1,3],[1]):
                    tokens=[states[slot].logits[:,-1].argmax().item() for slot in active]
                    actual=runtime.decode_batch(tokens,active)[0]
                    for row,(slot,token) in enumerate(zip(active,tokens)):
                        result=reference(torch.tensor([[token]]),past_key_values=states[slot].past_key_values,use_cache=True)
                        states[slot]=result
                        check(actual[row],result.logits[:,-1],'decode',slot)
            del runtime,model
            torch.cuda.empty_cache()
    print(json.dumps({'passed':True,'scope':'routed kernels and synthetic OLMoE variants','checks':checks},indent=2))


if __name__=='__main__':
    main()
