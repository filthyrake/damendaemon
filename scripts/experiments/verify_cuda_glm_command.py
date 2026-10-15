"""Owned CUDA serving subprocess gate: local snapshot, HTTP/SSE and SIGTERM."""
import argparse
import json
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
from urllib.request import Request,urlopen
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    family_options=parser.add_mutually_exclusive_group()
    family_options.add_argument('--flash',action='store_true')
    family_options.add_argument('--deepseek',action='store_true')
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    if args.deepseek:
        from tests.test_deepseek_generation import text_fixture
        from tests.test_deepseek_reference_loader import fixture
    elif args.flash:
        from tests.test_glm_flash_generation import text_fixture,snapshot
        def fixture(path,*,vocab_size):snapshot(path,text_fixture(path))
    else:
        from tests.test_glm_generation import text_fixture
        from tests.test_glm_reference_loader import fixture
    from tests.test_glm_flash_host_generation import zero_fixture_head
    family='deepseek_v4' if args.deepseek else 'glm5_next' if args.flash else 'glm_moe_dsa'
    report=dict(passed=False,family=family,zero_head_fixture=True,allocator_limit_mib=128,
                official_model_qualified=False,performance_qualified=False)
    save(args.output,report);process=None
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-cuda-command-') as temp:
            path=Path(temp);text=text_fixture(path);fixture(path,vocab_size=text.vocab_size)
            if args.deepseek:
                import struct
                shard=path/'model.safetensors';data=bytearray(shard.read_bytes())
                length=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+length])
                start,end=header['head.weight']['data_offsets'];data[8+length+start:8+length+end]=bytes(end-start)
                shard.write_bytes(data)
            else:zero_fixture_head(path)
            with tempfile.TemporaryFile(mode='w+') as errors:
                process=subprocess.Popen([sys.executable,'-m','scripts.serve_cuda_reference',str(path),
                    '--device','cuda:0','--port','0','--context','128','--weight-mib','1','--device-weight-mib','1',
                    '--allocator-mib','128','--cache-mib','1','--projection-workspace-mib','8' if args.deepseek else '1','--workspace-mib','256' if args.deepseek else '2','--output-mib','1'],
                    stdout=subprocess.PIPE,stderr=errors,text=True)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout,selectors.EVENT_READ)
                    if not selector.select(15):raise TimeoutError('CUDA command readiness deadline')
                line=process.stdout.readline()
                if not line:
                    errors.seek(0);raise RuntimeError('CUDA command startup failed: '+errors.read(4096))
                ready=json.loads(line)
                if ready.get('event')!='http_ready' or ready.get('scope')!='cuda_prepared_reference' or ready.get('device')!='cuda:0':
                    raise AssertionError('untruthful CUDA command readiness')
                base=ready['url'];reasoning=[];field='content' if args.deepseek else 'reasoning_content'
                for stream in (False,True):
                    body=dict(model='model',messages=[dict(role='user',content='Hi')],max_tokens=2,stream=stream)
                    if stream:body['stream_options']={'include_usage':True}
                    with urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=15) as response:
                        wire=response.read().decode()
                    if stream:
                        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
                        reasoning.append(''.join(c.get('delta',{}).get(field,'') for e in events for c in e['choices']))
                        if 'data: [DONE]' not in wire:raise AssertionError('missing SSE completion')
                        timing=events[-1]['dllm_timing']
                    else:
                        result=json.loads(wire);reasoning.append(result['choices'][0]['message'].get(field,''))
                        timing=result['dllm_timing']
                    if timing['source_tier']!='cuda_prepared_reference' or timing['kernel_backend']!=family+'-cuda-prepared-reference':
                        raise AssertionError('CPU timing label leaked into CUDA command')
                if not reasoning[0] or reasoning[0]!=reasoning[1]:raise AssertionError('chat/SSE control differs')
                process.terminate();stdout,_=process.communicate(timeout=5)
                events=[json.loads(line) for line in stdout.splitlines() if line.strip()]
                if process.returncode!=0 or not events or events[-1]!=dict(event='stopped',scope='cuda_prepared_reference'):
                    raise AssertionError('CUDA command failed owned signal shutdown')
                report.update(passed=True,chat=True,sse=True,sigterm_exit_code=process.returncode,stopped=events[-1],ready=ready)
    except BaseException as error:report['error']=str(error);raise
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.communicate(timeout=5)
        save(args.output,report)


if __name__=='__main__':main()
