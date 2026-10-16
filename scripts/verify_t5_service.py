"""Real HTTP qualification for raw T5/Mamba scheduling and residency."""
from dllm.evidence_environment import report_environment
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch
from dllm.bench import events
from dllm.models import load
from dllm.runtime import require_gpu
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--other-model',required=True)
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--chunk-size',type=int,default=0);p.add_argument('--prefix-budget',type=int,default=0)
    args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device
    require_gpu();torch.set_num_threads(4)
    model=load(args.model,512)
    encoder_decoder=model.config.get('is_encoder_decoder',False)
    prompt='The blue bird flies over the green field every morning.'
    if encoder_decoder:prompt='translate English to German: '+prompt
    ids=model.raw_tokens(prompt)
    runtime=BatchedRuntime({'a':args.model,'b':args.other_model},context=512,batch_size=4,gpu_models=1,
        prefill_chunk_size=args.chunk_size,prefix_cache_tokens=args.prefix_budget)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(body,path='/v1/completions'):
        with urlopen(Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
            return list(events(response, require_terminated=True)) if body.get('stream') else json.load(response)
    def body(prompt=ids,**kw):return {'model':'a','prompt':prompt,'max_tokens':8,'temperature':0,**kw}
    def rejected(payload,path='/v1/completions'):
        try:post(payload,path)
        except HTTPError as error:
            with error:
                if error.code != 400:
                    raise ValueError((error.code, error.read()))
                return
        raise AssertionError('invalid request accepted')
    try:
        expected=post(body())
        assert expected['usage']['prompt_tokens']==len(ids)
        assert post(body(prompt))['choices'][0]['text']==expected['choices'][0]['text']
        streamed=post(body(stream=True,stream_options={'include_usage':True}))
        assert streamed[-1]=='[DONE]'
        chunks=[json.loads(x) for x in streamed[:-1]]
        assert not any('error' in x for x in chunks)
        assert ''.join(c.get('text','') for x in chunks for c in x['choices'])==expected['choices'][0]['text']
        assert next(x['usage'] for x in chunks if x.get('usage'))==expected['usage']
        prompts=[ids,model.raw_tokens('summarize: A dog played in the park.'),ids*3,ids*5]
        serial=[post(body(x)) for x in prompts]
        with ThreadPoolExecutor(4) as pool:parallel=list(pool.map(lambda x:post(body(x)),prompts))
        assert [x['choices'][0]['text'] for x in parallel]==[x['choices'][0]['text'] for x in serial]
        # Encoders have independent context; causal models share prompt/output.
        post(body((ids*100)[:512 if encoder_decoder else 504],max_tokens=8))
        rejected(body([1]*513));rejected(body(max_tokens=513))
        rejected({'model':'a','messages':[{'role':'user','content':'hello'}]},'/v1/chat/completions')
        stream=runtime.generate('a',None,128,raw_prompt=ids)
        next(stream);stream.close();runtime.manage('status')
        assert post(body())['choices'][0]['text']==expected['choices'][0]['text']
        post({'model':'b','prompt':'The blue bird','max_tokens':8,'temperature':0})
        restored=post(body())
        assert restored['choices'][0]['text']==expected['choices'][0]['text']
        assert restored['dllm_timing']['source_tier']=='host',restored
        report={'passed':True,'device':args.device,'model':args.model,'other_model':args.other_model,
            'runtime_environment':report_environment(os.environ),
            'text_and_token_prompts_equal':True,'stream_and_nonstream_equal':True,'concurrent_isolation':True,
            'context_boundary_with_output':True,'encoder_decoder':encoder_decoder,
            'oversized_requests_rejected':True,'chat_rejected':True,
            'cancellation_recovery':True,'decoder_only_switch_and_host_restore':True,
            'restore_timing':restored['dllm_timing'],
            'prefill_chunk_size':args.chunk_size,'prefix_cache_tokens':args.prefix_budget,
            'sample':expected}
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()
    for options in (({'prefill_chunk_size':128},{'prefix_cache_tokens':512}) if encoder_decoder else ()):
        runtime=BatchedRuntime({'a':args.model},context=512,**options)
        try:
            try:list(runtime.generate('a',None,2,raw_prompt=ids))
            except ValueError as error:assert 'does not yet support' in str(error)
            else:raise AssertionError('unsupported cache option accepted')
        finally:runtime.close()
    report['unsupported_cache_options_rejected']=encoder_decoder
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
