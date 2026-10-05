"""Real image-chat HTTP, streaming, state isolation, and residency proof."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from unittest.mock import patch

from PIL import Image
import torch
from tokenizers import Tokenizer
from dllm.bench import events


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--output',required=True)
    parser.add_argument('--allocation-diagnostics',action='store_true')
    parser.add_argument('--warm-requests',type=int,default=3)
    parser.add_argument('--stage-profile',help='diagnostic stage JSONL path; adds GPU synchronization')
    parser.add_argument('--allocation-reference',help='trusted allocator trace identifying persistent cuBLAS workspace')
    args=parser.parse_args()
    if args.warm_requests<1 or (args.stage_profile and args.allocation_diagnostics):
        parser.error("positive warm request count and at most one diagnostic mode required")
    reference=torch.load(args.reference,weights_only=True)
    tokenizer=Tokenizer.from_file(str(Path(reference['model'])/'tokenizer.json'))
    tokenizer.no_padding();tokenizer.no_truncation()
    payloads=[]
    for case in reference['cases']:
        content=[]
        for image in case['images']:
            buffer=io.BytesIO();Image.fromarray(image.numpy()).save(buffer,format='PNG')
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+
                base64.b64encode(buffer.getvalue()).decode('ascii'))))
        content.append(dict(type='text',text='Describe the colors and shapes in these images.'))
        payloads.append(dict(model='a',messages=[dict(role='user',content=content)],
            max_tokens=8,temperature=0,chat_template_kwargs=dict(enable_thinking=False)))
    report=dict(passed=False,reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                stage_profile=bool(args.stage_profile),quantization=reference.get('text_quantization','none'),checks=[],trials=[],statuses={},
                torch=str(torch.__version__),device_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [Path('dllm')/name for name in ('hybrid.py','cuda_warmup.py','models.py','qwen2.py','runtime.py','scheduler.py','server.py',
                    'vision_policy.py','vision_model.py','vision_hybrid.py','vision_qwen35.py','vision_messages.py')]})
    from dllm.models import weight_bytes
    with patch.dict(os.environ,{'DLLM_EXPERIMENTAL_VISION':'1'}):
        report['estimated_host_weight_bytes']=weight_bytes(reference['model'],quantization=report['quantization'])
    workspace=0
    if args.allocation_reference:
        data=Path(args.allocation_reference).read_bytes()
        allocations=json.loads(data)
        if not allocations or any(not any('getNewWorkspace' in frame.get('name','')
                for frame in allocation.get('frames',[])) for allocation in allocations):
            raise ValueError('allocation reference contains unidentified live allocations')
        workspace=sum(allocation['size'] for allocation in allocations)
        report['library_allocation_reference_sha256']=hashlib.sha256(data).hexdigest()
    report['identified_persistent_cublas_workspace_bytes']=workspace
    with tempfile.TemporaryDirectory(prefix='dllm-vision-http-') as directory:
        logpath=Path(directory)/'server.log'
        env={**os.environ,'DLLM_EXPERIMENTAL_VISION':'1',
             'DLLM_QUANTIZATION':reference.get('text_quantization','none')}
        if args.stage_profile:env['DLLM_VISION_STAGE_PROFILE']=args.stage_profile
        with logpath.open('w') as log:
            command=([sys.executable,'scripts/diagnose_vision_server.py'] if args.allocation_diagnostics
                     else [sys.executable,'scripts/profile_vision_server.py'] if args.stage_profile
                     else [sys.executable,'-m','dllm'])
            process=subprocess.Popen([*command,'serve','--port','0',
                '--context','1024','--batch-size','4','--gpu-models','1','--host-models','2',
                '--host-gib','8','--model','a='+reference['model'],'--model','b='+reference['model']],
                env=env,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline and process.poll() is None:
                    lines=[line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                    if lines:base=lines[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                def open_request(path,body=None):
                    return urlopen(Request(base+path,data=None if body is None else json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'}),timeout=180)
                def request(body):
                    start=time.perf_counter()
                    with open_request('/v1/chat/completions',body) as response:result=json.load(response)
                    result['measured_http_seconds']=time.perf_counter()-start
                    return result
                def status(name):
                    with open_request('/v1/models/status') as response:value=json.load(response)
                    report['statuses'][name]=value
                    return value
                def check(result,index,label):
                    case=reference['cases'][index]
                    expected=tokenizer.decode(case['decode_tokens'],skip_special_tokens=True)
                    assert result['choices'][0]['message']['content']==expected,label
                    assert result['usage']['prompt_tokens']==len(case['tokens']),label
                    assert result['usage']['completion_tokens']==8,label
                    report['checks'].append(label)
                status('initial')
                for index,body in enumerate(payloads):
                    result=request(body);check(result,index,'independent:'+reference['cases'][index]['name'])
                    report['trials'].append(dict(name=reference['cases'][index]['name'],
                        http_seconds=result['measured_http_seconds'],timing=result['dllm_timing'],usage=result['usage']))
                loaded=status('loaded')
                assert loaded['models'][0]['vision_gpu_weight_bytes']==204977152
                assert loaded['models'][0]['host_weight_bytes']==report['estimated_host_weight_bytes']
                for _ in range(args.warm_requests):
                    result=request(payloads[0]);check(result,0,'warm')
                    report['trials'].append(dict(name='warm-square',http_seconds=result['measured_http_seconds'],
                        timing=result['dllm_timing'],usage=result['usage']))
                with ThreadPoolExecutor(4) as pool:
                    responses=list(pool.map(request,[payloads[i] for i in (0,3,2,5)]))
                for result,index in zip(responses,(0,3,2,5)):check(result,index,'concurrent:'+str(index))
                streamed=[];usage=None
                with open_request('/v1/chat/completions',{**payloads[0],'stream':True,
                        'stream_options':{'include_usage':True}}) as response:
                    for event in events(response):
                        if event=='[DONE]':break
                        chunk=json.loads(event)
                        assert not chunk.get('error'),chunk
                        for choice in chunk.get('choices',[]):streamed.append(choice.get('delta',{}).get('content',''))
                        if chunk.get('usage'):usage=chunk['usage']
                assert ''.join(streamed)==tokenizer.decode(reference['cases'][0]['decode_tokens'],skip_special_tokens=True)
                assert usage['prompt_tokens']==len(reference['cases'][0]['tokens'])
                report['checks'].append('streaming-content-and-usage')
                with open_request('/v1/chat/completions',{**payloads[0],'stream':True,'max_tokens':128}) as response:
                    for event in events(response):
                        if event=='[DONE]':raise AssertionError('stream ended before cancellation')
                        chunk=json.loads(event)
                        if any(c.get('delta',{}).get('content') for c in chunk.get('choices',[])):break
                check(request(payloads[3]),3,'disconnect-recovery-changed-image')
                for url in ('data:image/png;base64,???','data:image/png;base64,bm90LWEtcG5n',
                            'https://example.com/image.png'):
                    invalid={**payloads[0],'messages':[dict(role='user',content=[dict(type='image_url',image_url=dict(url=url))])]}
                    try:request(invalid)
                    except HTTPError as error:assert error.code==400
                    else:raise AssertionError('invalid image accepted')
                check(request(payloads[0]),0,'malformed-input-recovery')
                check(request({**payloads[0],'model':'b'}),0,'second-model-alias')
                restored=request(payloads[0]);check(restored,0,'host-restore')
                assert restored['dllm_timing']['source_tier']=='host'
                for alias in ('a','b'):
                    with open_request('/v1/models/unload',dict(model=alias,release_host=True)) as response:json.load(response)
                unloaded=status('unloaded')
                assert unloaded['host_cached_bytes']==0
                assert all(model['vision_gpu_weight_bytes']==0 for model in unloaded['models'])
                allowed=unloaded.get('diagnostic_cublas_workspace_bytes',workspace) if args.allocation_diagnostics else workspace
                assert unloaded['cuda_allocated_bytes'] is not None and unloaded['cuda_allocated_bytes']<=allowed,unloaded
                check(request(payloads[0]),0,'disk-reload-after-release')
                with open_request('/v1/models/unload',dict(model='a',release_host=True)) as response:json.load(response)
                again=status('unloaded-again')
                assert again['host_cached_bytes']==0 and again['cuda_allocated_bytes']==unloaded['cuda_allocated_bytes']
                report['checks'].append('repeat-release-no-allocation-growth')
                report['passed']=True
            except BaseException as error:
                report['error']=repr(error)
                print(logpath.read_text(),file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:process.wait(30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
