"""Inject an output-stage error during real batched distributed HTTP execution.

This qualifies failure propagation and cleanup, not automatic recovery or a
process crash. The production server and transport remain unmodified. An outer
owned process-group deadline covers NCCL teardown and stalled HTTP clients.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class InjectedStageFailure(RuntimeError):pass


def worker(args):
    from dllm.distributed_server_cli import main as serve
    options=[args.model,'--quantization','q8_0','--batch-size','4','--alias','a','--port','0']
    output=Path(args.output)
    if os.environ['RANK']=='1':
        from dllm.layer_stage import LayerStage
        method='run_prefill' if args.operation=='prefill' else 'run_decode'
        original=getattr(LayerStage,method)
        def fail_batch(stage,inputs,positions,slots):
            if args.operation=='prefill' or inputs.shape[0]==4:
                output.with_name(output.stem+'-worker.json').write_text(json.dumps(dict(
                    injected=True,operation=args.operation,rows=inputs.shape[0],rank=1,stage_layers=list(stage.layer_range)),indent=2)+'\n')
                raise InjectedStageFailure('injected output-stage batch failure')
            return original(stage,inputs,positions,slots)
        setattr(LayerStage,method,fail_batch)
        try:serve(options)
        except InjectedStageFailure:return
        finally:setattr(LayerStage,method,original)
        raise AssertionError('no batched output-stage failure was injected')
    from dllm.distributed_serving import DistributedRuntime
    from dllm.server import BoundedHTTPServer
    original_init=BoundedHTTPServer.__init__;original_run=DistributedRuntime.run
    base=None
    report=dict(passed=False,streaming=args.stream,scope='Real output-stage execution failure; no recovery/performance claim',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('dllm/distributed.py','dllm/distributed_serving.py','dllm/distributed_server_cli.py')})
    def capture(server,*positional,**keywords):
        nonlocal base
        original_init(server,*positional,**keywords);base=f'http://127.0.0.1:{server.server_port}'
    def client(runtime):
        barrier=threading.Barrier(4)
        def request(index):
            body=dict(model='a',prompt=[840,20772,1246,18495,3553,323,17179,1995,13,13655,279,16148,2797,323,63594,13],
                      max_tokens=32,temperature=0,stream=args.stream and index==0)
            barrier.wait(timeout=10)
            try:
                with urlopen(Request(base+'/v1/completions',data=json.dumps(body).encode(),
                             headers={'Content-Type':'application/json'}),timeout=45) as response:
                    if not body['stream']:raise AssertionError('failed batch returned success: '+response.read().decode())
                    from dllm.bench import events
                    chunks=list(events(response))
                    assert '[DONE]' not in chunks
                    parsed=[json.loads(chunk) for chunk in chunks]
                    assert any(chunk.get('error') for chunk in parsed)
                    assert not any(choice.get('finish_reason') for chunk in parsed for choice in chunk.get('choices',[]))
                    assert not any(chunk.get('usage') for chunk in parsed)
                    return dict(request=index,status=200,stream_error=True,chunks=parsed)
            except HTTPError as error:
                payload=json.load(error);error.close()
                assert error.code in (500,503) and payload.get('error')
                assert 'choices' not in payload and 'usage' not in payload
                return dict(request=index,status=error.code,payload=payload)
        try:
            with ThreadPoolExecutor(4) as pool:report['requests']=list(pool.map(request,range(4)))
            if args.stream and args.operation=='decode':assert any(item.get('stream_error') for item in report['requests'])
            assert runtime.closed.wait(2)
            status=runtime.manage('status');report['status']=status
            assert status['failed'] and status['closed'] and not status['active']
            assert status['completed']==0 and status['queued']==0 and runtime.pair.state.failed
            for path,body in (('/health',None),('/v1/completions',dict(model='a',prompt=[1],max_tokens=2))):
                try:
                    data=None if body is None else json.dumps(body).encode()
                    with urlopen(Request(base+path,data=data,headers={'Content-Type':'application/json'}),timeout=5):
                        raise AssertionError('closed service accepted '+path)
                except HTTPError as error:
                    payload=json.load(error);error.close();assert error.code==503
                    report.setdefault('closed_http_checks',[]).append(dict(path=path,status=error.code,payload=payload))
            report['passed']=True
        except BaseException as error:report['error']=repr(error)
        finally:runtime.close()
    def checked(runtime,*positional,**keywords):
        thread=threading.Thread(target=client,args=(runtime,),daemon=True);thread.start()
        try:return original_run(runtime,*positional,**keywords)
        except BaseException as error:
            report['coordinator_error_type']=type(error).__name__
            raise
        finally:
            runtime.close();thread.join(50)
            if thread.is_alive():report['passed']=False;report['error']='HTTP clients did not terminate'
            output.with_name(output.stem+'-coordinator.json').write_text(json.dumps(report,indent=2)+'\n')
    BoundedHTTPServer.__init__=capture;DistributedRuntime.run=checked
    try:
        try:serve(options)
        except Exception:
            if report['passed'] and report.get('coordinator_error_type'):return
            raise
    finally:BoundedHTTPServer.__init__=original_init;DistributedRuntime.run=original_run
    raise AssertionError('coordinator did not report an execution/transport failure')


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--stream',action='store_true')
    p.add_argument('--operation',choices=('prefill','decode'),default='decode');args=p.parse_args()
    if args.worker:return worker(args)
    from distributed_job import run_job
    output=Path(args.output)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node','2',
        '--run-path',str(Path(__file__).resolve()),args.model,'--output',str(output),'--operation',args.operation,'--worker']
    if args.stream:command.append('--stream')
    try:result=run_job(command,timeout=180)
    except subprocess.CalledProcessError as error:
        print(error.stdout,file=sys.stderr);print(error.stderr,file=sys.stderr);raise
    coordinator=json.loads(output.with_name(output.stem+'-coordinator.json').read_text())
    peer=json.loads(output.with_name(output.stem+'-worker.json').read_text())
    assert coordinator['passed'] and peer['injected'] and (peer['operation']=='prefill' or peer['rows']>1)
    output.write_text(json.dumps(dict(passed=True,coordinator=coordinator,worker=peer,
                                     owned_job_exit_code=result.returncode),indent=2)+'\n')
    print(json.dumps(dict(passed=True,failed_http_requests=len(coordinator['requests']),
                         injected_operation=peer['operation'],rows=peer['rows'])),flush=True)


if __name__=='__main__':main()
