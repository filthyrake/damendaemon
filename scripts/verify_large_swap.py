"""Prove one-resident-model swaps when two executors cannot coexist in VRAM."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.request import Request,urlopen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='dllm-large-swap-') as directory:
        logpath=Path(directory)/'server.log'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--context','2048',
                '--gpu-models','1','--host-models','2','--host-gib','20','--model','a='+args.a,'--model','b='+args.b],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline and process.poll() is None:
                    ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                    if ready:base=ready[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                reports=[]
                for alias in ('a','b','a'):
                    body={'model':alias,'messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}], 'max_tokens':8}
                    with urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as response:
                        result=json.load(response)
                    assert result['choices'][0]['message']['content']
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                    memory=[line.strip().split(', ') for line in apps.splitlines() if line.strip().split(', ')[1]==str(process.pid)]
                    assert len(memory)==1 and memory[0][0]=='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
                    reports.append({'model':alias,'memory_mib':int(memory[0][2]),'response':result})
                assert [r['response']['dllm_timing']['source_tier'] for r in reports]==['disk','disk','host']
                assert reports[0]['response']['choices']==reports[2]['response']['choices']
                total=int(subprocess.check_output(['nvidia-smi','--id=GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
                assert reports[0]['memory_mib']+reports[1]['memory_mib']>total
                print(json.dumps({'passed':True,'models':{'a':args.a,'b':args.b},'gpu_total_mib':total,'reports':reports},indent=2))
            except Exception:
                print(logpath.read_text(),file=sys.stderr);raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
