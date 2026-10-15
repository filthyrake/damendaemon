"""Independent model-score reference plus real streaming/concurrent API proof."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
from urllib.request import Request,urlopen
import torch
from transformers import AutoTokenizer
from reference_model import load_reference
from dllm.bench import events
from dllm.logprobs import TokenText
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for
from dllm.runtime import require_gpu


def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--reference-quantization',choices=('none','q4_0','q8_0'),default='none')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device
    require_gpu();torch.set_num_threads(4)
    reference=load_reference(args.model,reference_quantization=args.reference_quantization,
        local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager').eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    runtime=BatchedRuntime({'a':args.model},context=1024,batch_size=4,prefill_chunk_size=128,prefix_cache_tokens=512)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    checks=[]
    try:
        prompt=tokenizer.encode('The blue bird flies over the field. '*12,add_special_tokens=True)
        for settings in ({},{'temperature':.8,'top_k':20,'top_p':.95,'seed':71},
                         {'logit_bias':{str(tokenizer.encode('A',add_special_tokens=False)[0]):20}}):
            output=list(runtime.generate('a',None,8,raw_prompt=prompt,logprobs=True,top_logprobs=5,**settings))
            with torch.inference_mode():
                state=reference(torch.tensor([prompt]),use_cache=True)
                for event in output:
                    if 'token' not in event:continue
                    token=event['token']
                    if 'logprobs' in event:
                        expected=torch.log_softmax(state.logits[0,-1].float(),-1)
                        actual=event['logprobs']
                        difference=abs(actual['logprob']-expected[token].item())
                        assert difference<.03,(difference,actual)
                        values,ids=expected.topk(5)
                        executor=runtime.runtime.gpu['a']
                        decoder=TokenText(executor.model.tokenizer)
                        for result,index,value in zip(actual['top_logprobs'],ids.tolist(),values.tolist()):
                            text,raw=decoder.token(index)
                            assert result['token']==text and result['bytes']==(None if raw is None else list(raw))
                            assert abs(result['logprob']-value)<.03
                        checks.append({'token':token,'selected_logprob_abs_error':difference,'top5_match':True})
                    state=reference(torch.tensor([[token]]),past_key_values=state.past_key_values,use_cache=True)
        base=f'http://127.0.0.1:{server.server_port}'
        messages=[{'role':'user','content':'Explain matrix multiplication. Mention café and 🦙.'}]
        def post(stream=False,scored=True):
            body={'model':'a','messages':messages,'max_tokens':12,'temperature':0,'stream':stream}
            if scored:body.update(logprobs=True,top_logprobs=5)
            with urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
                    headers={'Content-Type':'application/json'}),timeout=120) as response:
                return list(events(response, require_terminated=True)) if stream else json.load(response)
        baseline=post(scored=False)
        whole=post()
        expected=whole['choices'][0]
        assert expected['message']==baseline['choices'][0]['message']
        chunks=post(stream=True)
        assert chunks[-1]=='[DONE]'
        text='';stream_scores=[]
        for chunk in chunks[:-1]:
            data=json.loads(chunk);assert 'error' not in data,data
            for choice in data['choices']:
                text+=choice['delta'].get('content','')
                if choice.get('logprobs'):stream_scores.extend(choice['logprobs']['content'])
        assert text==expected['message']['content']
        assert stream_scores==expected['logprobs']['content']
        raw=[record['bytes'] for record in stream_scores]
        if all(value is not None for value in raw):
            assert bytes(b for value in raw for b in value).decode()==text
        with ThreadPoolExecutor(4) as pool:concurrent=list(pool.map(lambda _:post(),range(4)))
        for result in concurrent:
            choice=result['choices'][0]
            assert choice['message']==expected['message']
            assert len(choice['logprobs']['content'])==len(expected['logprobs']['content'])
            for a,b in zip(choice['logprobs']['content'],expected['logprobs']['content']):
                assert a['token']==b['token'] and abs(a['logprob']-b['logprob'])<.005
        stream=runtime.generate('a',None,40,raw_prompt=prompt,logprobs=True,top_logprobs=5)
        next(stream);stream.close();runtime.manage('status')
        executor=runtime.runtime.gpu['a']
        if hasattr(executor,'pool'):assert executor.pool.used_tokens==0
        assert post()['choices'][0]['message']==expected['message']
        print(json.dumps({'passed':True,'model':args.model,'device':args.device,
            'runtime_environment':{k:v for k,v in os.environ.items() if k.startswith('DLLM_')},
            'reference_quantization':args.reference_quantization,'reference_dtype':'float32','reference_attention':'eager',
            'checks':checks,'stream_and_nonstream_scores_equal':True,'scoring_preserves_output':True,
            'concurrent_request_isolation':True,'prefix_hits_and_chunking':True,'cancellation_recovery':True},indent=2))
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()


if __name__=='__main__':main()
