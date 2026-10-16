"""Live empty-server registration, residency commands and active-stream fairness."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from dllm.bench import events


def main():
    p=argparse.ArgumentParser()
    p.add_argument('a',type=Path)
    p.add_argument('b',type=Path)
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=p.parse_args()
    a,b=str(args.a.resolve()),str(args.b.resolve())
    with tempfile.TemporaryDirectory(prefix='dllm-management-') as directory:
        logpath=Path(directory)/'server.log'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--device',args.device,
                '--context','512','--gpu-models','1','--host-models','2'],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline and process.poll() is None:
                    ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                    if ready:base=ready[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                def post(route,body):
                    return urlopen(Request(base+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180)
                def manage(action,**body):
                    with post('/v1/models/'+action,body) as response:return json.load(response)
                def status():
                    with urlopen(base+'/v1/models/status',timeout=180) as response:return json.load(response)
                def tier(result,alias='a'):
                    return next(m['residency'] for m in result['models'] if m['id']==alias)
                prompt={'model':'a','messages':[{'role':'user','content':'Explain matrix multiplication in detail.'}],'max_tokens':96}
                def generate(alias='a'):
                    with post('/v1/chat/completions',{**prompt,'model':alias}) as response:return json.load(response)
                assert status()['models']==[]
                for _ in range(2):assert tier(manage('register',model='a',path=a))=='disk'
                assert manage('load',model='a')['source_tier']=='disk'
                assert manage('load',model='a')['source_tier']=='gpu'
                first=generate()
                assert tier(manage('unload',model='a'))=='host'
                assert manage('load',model='a')['source_tier']=='host'
                for _ in range(2):
                    cold=manage('unload',model='a',release_host=True)
                    assert tier(cold)=='disk' and cold['host_cached_bytes']==0
                assert args.a.is_file() or (args.a/'config.json').is_file()
                manage('load',model='a')
                with post('/v1/chat/completions',{**prompt,'stream':True,'stream_options':{'include_usage':True}}) as response:
                    stream=events(response, require_terminated=True)
                    chunks=[]
                    for item in stream:
                        chunks.append(item)
                        if any(c.get('delta',{}).get('content') for c in json.loads(item)['choices']):break
                    with ThreadPoolExecutor(1) as pool:
                        unloaded=pool.submit(manage,'unload',model='a')
                        chunks.extend(stream)
                        assert tier(unloaded.result(180))=='host'
                assert chunks[-1]=='[DONE]'
                text=''.join(c.get('delta',{}).get('content','') for item in chunks[:-1] for c in json.loads(item)['choices'])
                assert text==first['choices'][0]['message']['content']
                restored=generate()
                assert restored['dllm_timing']['source_tier']=='host' and restored['choices']==first['choices']
                added=json.loads(subprocess.check_output([sys.executable,'-m','dllm','add','b',b,'--server',base],text=True,timeout=180))
                assert tier(added,'b')=='disk'
                for action,body in (('register',{'model':'bad','path':str(Path(directory)/'missing')}),
                                    ('register',{'model':'a','path':b}),('load',{'model':'unknown'})):
                    try:manage(action,**body);raise AssertionError('invalid management command accepted')
                    except HTTPError as error:
                        with error:
                            if error.code != 400:
                                raise ValueError('unexpected HTTP rejection status')
                assert generate('b')['choices'][0]['message']['content']
                assert generate()['choices']==first['choices']
                listing=json.loads(subprocess.check_output([sys.executable,'-m','dllm','models','--server',base],text=True,timeout=180))
                assert {m['id'] for m in listing['models']}=={'a','b'}
                cli_unload=json.loads(subprocess.check_output([sys.executable,'-m','dllm','unload','a','--release-host','--server',base],text=True,timeout=180))
                assert tier(cli_unload)=='disk'
                cli_load=json.loads(subprocess.check_output([sys.executable,'-m','dllm','load','a','--server',base],text=True,timeout=180))
                assert tier(cli_load)=='gpu'
                print(json.dumps({'passed':True,'device':args.device,'models':{'a':a,'b':b},
                    'empty_registry':True,'idempotent_registration':True,'host_and_gpu_unload':True,
                    'active_stream_unload':True,'cli_management':True,'error_recovery':True,
                    'disk_preserved':True,'final_status':cli_load},indent=2))
            except Exception:
                print(logpath.read_text(),file=sys.stderr);raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
