"""Bounded two-process native CUDA host HTTP/SSE and acknowledged signal shutdown."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--family',choices=('glm','flash','deepseek'),required=True)
    parser.add_argument('--images',action='store_true')
    args=parser.parse_args();require_guard()
    if args.images and args.family!='flash':raise ValueError('images require Flash')
    if args.output.exists():raise ValueError('new report required')
    if args.images:
        from tests.test_glm_flash_inline_images import snapshot,messages as image_messages
        from tests.test_glm_flash_image_generation import text_fixture
        def fixture(path,*,vocab_size):snapshot(path)
        ranges=[[0,1],[1,3]]
    elif args.family=='glm':
        from tests.test_glm_generation import text_fixture
        from tests.test_glm_reference_loader import fixture
        ranges=[[0,2],[2,3]]
    elif args.family=='flash':
        from tests.test_glm_flash_generation import text_fixture,snapshot
        def fixture(path,*,vocab_size):snapshot(path,text_fixture(path))
        ranges=[[0,1],[1,3]]
    else:
        from tests.test_deepseek_generation import text_fixture
        from tests.test_deepseek_reference_loader import fixture
        ranges=[[0,1],[1,3]]
    from tests.test_glm_flash_host_generation import zero_fixture_head
    from tests.tls_credentials import credentials
    from damendaemon.cpu_snapshot_identity import fingerprint
    report=dict(passed=False,family=args.family,zero_head_fixture=True,allocator_mib_per_process=128,
                max_cuda_processes=2,images=args.images,physical_multi_gpu_qualified=False,performance_qualified=False)
    save(args.output,report);processes=[]
    try:
        with tempfile.TemporaryDirectory(prefix='damendaemon-cuda-host-command-') as temp,tempfile.TemporaryFile(mode='w+') as errors:
            root=Path(temp);path=root/'model';path.mkdir();text=text_fixture(path);fixture(path,vocab_size=text.vocab_size)
            if args.family=='deepseek':
                import struct
                shard=path/'model.safetensors';data=bytearray(shard.read_bytes());length=struct.unpack('<Q',data[:8])[0]
                header=json.loads(data[8:8+length]);start,end=header['head.weight']['data_offsets']
                data[8+length+start:8+length+end]=bytes(end-start);shard.write_bytes(data)
            else:zero_fixture_head(path)
            cert=credentials();plan=root/'plan.json';plan.write_text(json.dumps(dict(ranges=ranges)))
            identity=fingerprint(path).model_id;peers=root/'peers.json'
            common=[str(path),'--family',args.family,'--native-chat','--device','cuda:0','--allocator-mib','128',
                '--device-weight-mib','1','--transfer-mib','1','--device-workspace-mib','256','--weight-mib','1',
                '--cache-mib','1','--output-mib','1','--plan',str(plan),'--model-id',identity,'--verify-model-id',
                '--ca',str(cert/'ca.pem'),'--context','128','--context-limit','128','--slots','2','--prefill-chunk','16',
                '--max-new-tokens','2','--timeout','10','--heartbeat-interval','1']+(['--flash-images'] if args.images else [])
            def launch(role,extra):
                process=subprocess.Popen([sys.executable,'-m','scripts.run_cuda_host_topology',role,*common,*extra],
                    stdout=subprocess.PIPE,stderr=errors,text=True);processes.append(process);return process
            def event(process):
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout,selectors.EVENT_READ)
                    if not selector.select(15):raise TimeoutError('CUDA host readiness deadline')
                line=process.stdout.readline()
                if not line:
                    errors.seek(0);raise RuntimeError('CUDA host startup failed: '+errors.read(4096))
                return json.loads(line)
            worker=launch('worker',['--index','1','--address','127.0.0.1','--port','0','--cert',str(cert/'server.pem'),
                '--key',str(cert/'server.key'),'--peer-cert',str(cert/'client.pem')])
            listening=event(worker)
            if listening.get('event')!='listening':raise AssertionError('worker did not listen')
            peers.write_text(json.dumps([dict(stage=1,address='127.0.0.1',port=listening['port'],server_name='localhost',peer_cert=str(cert/'server.pem'))]))
            coordinator=launch('serve',['--peers',str(peers),'--http-port','0','--cert',str(cert/'client.pem'),'--key',str(cert/'client.key')]+(['--vision-weight-mib','1','--image-workspace-mib','16','--image-output-mib','1','--max-patches','64'] if args.images else []))
            ready=event(coordinator);worker_ready=event(worker)
            if (ready.get('event')!='http_ready' or ready.get('execution')!='host_topology' or ready.get('local_device')!='cuda:0'
                    or ready.get('peer_devices_verified') is not False or worker_ready.get('local_device')!='cuda:0'):
                raise AssertionError('CUDA host placement readiness differs')
            if args.images and (ready.get('vision_device')!='cpu' or ready.get('modality')!='still_image'):
                raise AssertionError('incorrect CPU vision/CUDA language readiness')
            reasoning=[];field='content' if args.family=='deepseek' else 'reasoning_content'
            for stream in (False,True):
                body=dict(model='model',messages=image_messages() if args.images else [dict(role='user',content='Hi')],max_tokens=2,stream=stream)
                if stream:body['stream_options']={'include_usage':True}
                with urlopen(Request(ready['url']+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=15) as response:
                    wire=response.read().decode()
                if stream:
                    events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
                    reasoning.append(''.join(c.get('delta',{}).get(field,'') for e in events for c in e['choices']))
                    if 'data: [DONE]' not in wire:raise AssertionError('incomplete host SSE')
                    timing=events[-1]['damendaemon_timing']
                else:
                    result=json.loads(wire);reasoning.append(result['choices'][0]['message'].get(field,''));timing=result['damendaemon_timing']
                if timing['source_tier']!='host_reference' or timing['local_device']!='cuda:0' or timing['peer_devices_verified']:
                    raise AssertionError('incorrect host timing placement')
            if not reasoning[0] or reasoning[0]!=reasoning[1]:raise AssertionError('host chat/SSE content differs')
            with urlopen(ready['url']+'/v1/models/status',timeout=5) as response:status=json.load(response)
            if status['device']!='distributed' or status['local_device']!='cuda:0' or not status['model_identity_verified']:
                raise AssertionError('incorrect distributed host status')
            coordinator.terminate();stdout,_=coordinator.communicate(timeout=5);worker_out,_=worker.communicate(timeout=5)
            stopped=[json.loads(line) for line in stdout.splitlines() if line.strip()]
            if coordinator.returncode or worker.returncode or not stopped or not stopped[-1].get('graceful_stop') or '"event": "stopped"' not in worker_out:
                raise AssertionError('CUDA host command failed acknowledged STOP/cleanup')
            report.update(passed=True,chat=True,sse=True,ready=ready,worker_ready=worker_ready,
                sigterm_exit_code=coordinator.returncode,worker_exit_code=worker.returncode,stopped=stopped[-1])
    except BaseException as error:report['error']=str(error);raise
    finally:
        for process in processes:
            if process.poll() is None:process.kill()
        for process in processes:process.communicate(timeout=5)
        save(args.output,report)


if __name__=='__main__':main()
