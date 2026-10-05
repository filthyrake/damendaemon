"""Keep failed gates visible while isolating packed kernels from decoder precision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('hf_config')
    p.add_argument('--teacher',required=True);p.add_argument('--prefix',required=True);args=p.parse_args()
    results=[]
    for name,options,native in (
            ('triton',[],False),('expanded',['--expanded-control'],False),
            ('float32-reference',['--reference-dtype','float32'],True)):
        environment=os.environ.copy()
        if not native:environment.pop('DLLM_KQUANT_CUDA_LIBRARY',None)
        output=Path(args.prefix+'-'+name+'.json');log=output.with_suffix('.log')
        command=[sys.executable,'scripts/verify_gguf_reference.py',args.model,args.hf_config,
            '--output',str(output),'--teacher-trace',args.teacher,*options]
        print(f'Checking {name}',file=sys.stderr,flush=True)
        with log.open('w') as handle:
            run=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
        results.append(dict(case=name,exit_code=run.returncode,report=str(output),log=str(log)))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
