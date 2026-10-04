"""Synthetic architecture gate for bias/tied-head/window variants; not a checkpoint qualification."""
import json
from pathlib import Path
import tempfile
import torch
from tokenizers import Tokenizer, models
from transformers import Starcoder2Config, Starcoder2ForCausalLM
from dllm.qwen2 import ModelWeights
from dllm.runtime import make_executor, require_gpu


def main():
    require_gpu()
    torch.set_num_threads(4)
    checks=[]
    for biased,tied in ((True,False),(False,True)):
        torch.manual_seed(19)
        config=Starcoder2Config(vocab_size=256,hidden_size=64,intermediate_size=192,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=256,sliding_window=32,use_bias=biased,
            tie_word_embeddings=tied,bos_token_id=0,eos_token_id=1)
        config._attn_implementation='eager'
        reference=Starcoder2ForCausalLM(config).float().eval()
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            tokenizer=Tokenizer(models.WordLevel({str(i):i for i in range(256)},unk_token='0'))
            tokenizer.save(str(Path(directory)/'tokenizer.json'))
            (Path(directory)/'tokenizer_config.json').write_text('{}')
            weights=ModelWeights(directory,128)
            runtime=make_executor(weights,max_batch=4)
            prompts=[list(range(3,3+n)) for n in (31,32,33,65)]
            slots=[3,0,2,1]
            with torch.inference_mode():
                actual=runtime.prefill_batch(prompts,slots)
                states={}
                for row,(ids,slot) in enumerate(zip(prompts,slots)):
                    result=reference(torch.tensor([ids]),use_cache=True)
                    states[slot]=result
                    target=result.logits[:,-1]
                    relative=((actual[row].float().cpu()-target).square().mean()/target.square().mean()).sqrt().item()
                    assert relative<.01
                    checks.append(dict(bias=biased,tied=tied,phase='prefill',length=len(ids),relative_rms=relative))
                for active in ([1,3,0,2],[2,1],[3], [3,2,1]):
                    tokens=[states[slot].logits[:,-1].argmax().item() for slot in active]
                    actual=runtime.decode_batch(tokens,active)[0]
                    for row,(slot,token) in enumerate(zip(active,tokens)):
                        result=reference(torch.tensor([[token]]),past_key_values=states[slot].past_key_values,use_cache=True)
                        states[slot]=result
                        target=result.logits[:,-1]
                        relative=((actual[row].float().cpu()-target).square().mean()/target.square().mean()).sqrt().item()
                        assert relative<.01
                        checks.append(dict(bias=biased,tied=tied,phase='decode',relative_rms=relative))
            del runtime,weights
            torch.cuda.empty_cache()
    print(json.dumps({'passed':True,'scope':'synthetic StarCoder2 architecture variants','checks':checks},indent=2))


if __name__=='__main__':
    main()
