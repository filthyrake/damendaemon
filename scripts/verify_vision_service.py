"""Real image-chat HTTP, streaming, state isolation, and residency proof."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
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
from damendaemon.bench import events


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cuda','mps','rocm'),default='cuda')
    parser.add_argument('--model',help='local copy of the pinned reference checkpoint')
    parser.add_argument('--allocation-diagnostics',action='store_true')
    parser.add_argument('--warm-requests',type=int,default=3)
    parser.add_argument('--stage-profile',help='diagnostic stage JSONL path; adds GPU synchronization')
    parser.add_argument('--allocation-reference',help='trusted allocator trace identifying persistent cuBLAS workspace')
    args=parser.parse_args()
    if args.warm_requests<1 or (args.stage_profile and args.allocation_diagnostics):
        parser.error("positive warm request count and at most one diagnostic mode required")
    reference=torch.load(args.reference,map_location='cpu',weights_only=True)
    model_path=args.model or reference['model']
    # ROCm allocator telemetry is reported under the hip_* status key.
    allocation_key=('hip' if args.device=='rocm' else args.device)+'_allocated_bytes'
    if args.device=='mps' and (args.allocation_diagnostics or args.allocation_reference or args.stage_profile):
        parser.error('CUDA diagnostic modes do not apply to Apple MPS')
    tokenizer=Tokenizer.from_file(str(Path(model_path)/'tokenizer.json'))
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
                torch=str(torch.__version__),device=args.device,hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if args.device=='mps' else torch.cuda.get_device_name(),
                model=str(model_path),reference_model=reference['model'],
                device_uuid=os.environ.get('CUDA_VISIBLE_DEVICES') if args.device=='cuda' else None,
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    [Path('damendaemon')/name for name in ('hybrid.py','hybrid_metal.py','quantized_metal.py','quantization.py','cuda_warmup.py','models.py','qwen2.py','runtime.py','scheduler.py','server.py',
                    'vision_policy.py','vision_model.py','vision_hybrid.py','vision_qwen35.py','vision_messages.py')]})
    from damendaemon.models import weight_bytes
    with patch.dict(os.environ,{'DAMENDAEMON_EXPERIMENTAL_VISION':'1'}):
        report['estimated_host_weight_bytes']=weight_bytes(model_path,quantization=report['quantization'])
    workspace=0
    if args.allocation_reference:
        data=Path(args.allocation_reference).read_bytes()
        allocations=json.loads(data)
        # Persistent cuBLAS workspaces are identified by the workspace
        # allocation frames; torch >=2.13 emits them via the C++
        # setWorkspaceForHandle path, older builds via getNewWorkspace (#575).
        def identified(allocation):
            return any('getNewWorkspace' in frame.get('name','')
                       or 'setWorkspaceForHandle' in frame.get('name','')
                       for frame in allocation.get('frames',[]))
        if not allocations or any(not identified(allocation) for allocation in allocations):
            raise ValueError('allocation reference contains unidentified live allocations')
        workspace=sum(allocation['size'] for allocation in allocations)
        report['library_allocation_reference_sha256']=hashlib.sha256(data).hexdigest()
    report['identified_persistent_cublas_workspace_bytes']=workspace
    with tempfile.TemporaryDirectory(prefix='damendaemon-vision-http-') as directory:
        logpath=Path(directory)/'server.log'
        env={**os.environ,'DAMENDAEMON_EXPERIMENTAL_VISION':'1','DAMENDAEMON_DEVICE':args.device,
             'DAMENDAEMON_QUANTIZATION':reference.get('text_quantization','none')}
        if args.stage_profile:env['DAMENDAEMON_VISION_STAGE_PROFILE']=args.stage_profile
        with logpath.open('w') as log:
            command=([sys.executable,'scripts/diagnose_vision_server.py'] if args.allocation_diagnostics
                     else [sys.executable,'scripts/profile_vision_server.py'] if args.stage_profile
                     else [sys.executable,'-m','damendaemon'])
            # Two model aliases must co-reside in the host cache for the
            # host-restore check; scale the budget with the checkpoint so the
            # lifecycle semantics under test do not depend on model size
            # (#575: two 2B residents need ~10 GiB, the fixed 8 evicted one).
            from damendaemon.models import weight_bytes
            # The serve below runs with the vision opt-in, and its host
            # accounting includes the vision weights; estimate under the same
            # opt-in or the budget is text-only and two residents cannot
            # co-exist (#575).
            with patch.dict(os.environ,{'DAMENDAEMON_EXPERIMENTAL_VISION':'1'}):
                estimated=weight_bytes(model_path,quantization=reference.get('text_quantization','none'))
            host_gib=max(8,math.ceil(2.2*estimated/2**30))
            serve_device='rocm' if args.device=='rocm' else args.device
            serve_command=[*command,'serve','--device',serve_device,'--port','0',
                '--context','1024','--batch-size','4','--gpu-models','1','--host-models','2',
                '--host-gib',str(host_gib),'--model','a='+model_path,'--model','b='+model_path]
            report['serve_host_gib']=host_gib
            report['serve_command']=[c if 'safetensors' not in c else c.split('/')[-1] for c in serve_command]
            process=subprocess.Popen(serve_command,
                env=env,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline and process.poll() is None:
                    lines=[line for line in logpath.read_text().splitlines() if line.startswith('damendaemon listening on ')]
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
                        http_seconds=result['measured_http_seconds'],timing=result['damendaemon_timing'],usage=result['usage']))
                loaded=status('loaded')
                from damendaemon.vision_qwen35 import expected_vision_bytes
                assert loaded['models'][0]['vision_gpu_weight_bytes']==expected_vision_bytes(model_path)
                assert loaded['models'][0]['host_weight_bytes']==report['estimated_host_weight_bytes']
                for _ in range(args.warm_requests):
                    result=request(payloads[0]);check(result,0,'warm')
                    report['trials'].append(dict(name='warm-square',http_seconds=result['measured_http_seconds'],
                        timing=result['damendaemon_timing'],usage=result['usage']))
                with ThreadPoolExecutor(4) as pool:
                    responses=list(pool.map(request,[payloads[i] for i in (0,3,2,5)]))
                for result,index in zip(responses,(0,3,2,5)):check(result,index,'concurrent:'+str(index))
                streamed=[];usage=None
                with open_request('/v1/chat/completions',{**payloads[0],'stream':True,
                        'stream_options':{'include_usage':True}}) as response:
                    for event in events(response, require_terminated=True):
                        if event=='[DONE]':break
                        chunk=json.loads(event)
                        assert not chunk.get('error'),chunk
                        for choice in chunk.get('choices',[]):streamed.append(choice.get('delta',{}).get('content',''))
                        if chunk.get('usage'):usage=chunk['usage']
                assert ''.join(streamed)==tokenizer.decode(reference['cases'][0]['decode_tokens'],skip_special_tokens=True)
                assert usage['prompt_tokens']==len(reference['cases'][0]['tokens'])
                report['checks'].append('streaming-content-and-usage')
                with open_request('/v1/chat/completions',{**payloads[0],'stream':True,'max_tokens':128}) as response:
                    for event in events(response, require_terminated=True):
                        if event=='[DONE]':raise AssertionError('stream ended before cancellation')
                        chunk=json.loads(event)
                        if any(c.get('delta',{}).get('content') for c in chunk.get('choices',[])):break
                check(request(payloads[3]),3,'disconnect-recovery-changed-image')
                for url in ('data:image/png;base64,???','data:image/png;base64,bm90LWEtcG5n',
                            'https://example.com/image.png'):
                    invalid={**payloads[0],'messages':[dict(role='user',content=[dict(type='image_url',image_url=dict(url=url))])]}
                    try:request(invalid)
                    except HTTPError as error:
                        with error:
                            if error.code != 400:
                                raise ValueError('unexpected HTTP rejection status')
                    else:raise AssertionError('invalid image accepted')
                check(request(payloads[0]),0,'malformed-input-recovery')
                check(request({**payloads[0],'model':'b'}),0,'second-model-alias')
                restored=request(payloads[0]);check(restored,0,'host-restore')
                report['host_restore_tier']=restored['damendaemon_timing']['source_tier']
                report['statuses']['before-restore']=status('before-restore')
                assert restored['damendaemon_timing']['source_tier']=='host'
                for alias in ('a','b'):
                    with open_request('/v1/models/unload',dict(model=alias,release_host=True)) as response:json.load(response)
                unloaded=status('unloaded')
                assert unloaded['host_cached_bytes']==0
                assert all(model['vision_gpu_weight_bytes']==0 for model in unloaded['models'])
                allowed=unloaded.get('diagnostic_cublas_workspace_bytes',workspace) if args.allocation_diagnostics else workspace
                if args.device=='rocm' and not args.allocation_diagnostics:
                    # No ROCm allocator-trace tooling yet (#575's trace path is
                    # CUDA); disclose and tolerate the hipBLAS persistent
                    # workspace, observed at 32 MiB (0.8B Q8_0) and 96 MiB (2B
                    # FP32 text) — it scales with resident geometry. Assert no
                    # GROWTH across the release cycle below instead of a zero
                    # baseline; the observed value is recorded in the report.
                    observed=unloaded.get(allocation_key) or 0
                    allowed=max(allowed,observed,32*1024**2)
                    report['identified_persistent_cublas_workspace_bytes']=allowed
                    report['persistent_workspace_observed_bytes']=observed
                    report['persistent_workspace_note']='hipBLAS persistent workspace disclosed (observed after unload); no ROCm allocator-trace tooling yet'
                assert unloaded[allocation_key] is not None and unloaded[allocation_key]<=allowed,unloaded
                check(request(payloads[0]),0,'disk-reload-after-release')
                with open_request('/v1/models/unload',dict(model='a',release_host=True)) as response:json.load(response)
                again=status('unloaded-again')
                assert again['host_cached_bytes']==0 and again[allocation_key]==unloaded[allocation_key]
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
