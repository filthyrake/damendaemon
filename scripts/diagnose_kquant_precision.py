"""Keep failed gates visible while isolating packed kernels from decoder precision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('hf_config')
    p.add_argument('--teacher',required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--cases',nargs='+',choices=('triton','expanded','float32-reference','standard'),
                   default=['triton','expanded','float32-reference']);args=p.parse_args()
    results=[]
    cases={'triton':([],False),'expanded':(['--expanded-control'],False),
           'float32-reference':(['--reference-dtype','float32'],True),'standard':([],True)}
    for name in args.cases:
        options,native=cases[name]
        environment=os.environ.copy()
        environment.pop('DAMENDAEMON_KQUANT_RESIDUAL_LIBRARY',None)
        if name=='standard':environment['DAMENDAEMON_LLAMA_PRECISION']='standard'
        if not native:environment.pop('DAMENDAEMON_KQUANT_CUDA_LIBRARY',None)
        output=Path(args.prefix+'-'+name+'.json');log=output.with_suffix('.log')
        command=[sys.executable,'scripts/verify_gguf_reference.py',args.model,args.hf_config,
            '--output',str(output),'--teacher-trace',args.teacher,'--context',str(args.context),*options]
        print(f'Checking {name}',file=sys.stderr,flush=True)
        with log.open('w') as handle:
            run=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
        results.append(dict(case=name,exit_code=run.returncode,report=str(output),log=str(log)))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
