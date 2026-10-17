"""Independent CPU recurrence and untouched-slot proof for the Metal delta scan."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
from pathlib import Path
import torch
from damendaemon.hybrid_metal import scan


def cpu_scan(q,k,v,decay,beta,state,slots,reset):
    selected=torch.zeros_like(state[slots]) if reset else state[slots].clone()
    outputs=[]
    for t in range(q.shape[1]):
        selected=selected*decay[:,t].exp()[...,None,None]
        delta=(v[:,t]-(selected*k[:,t,:,:,None]).sum(-2))*beta[:,t,:,None]
        selected=selected+k[:,t,:,:,None]*delta.unsqueeze(-2)
        outputs.append((selected*q[:,t,:,:,None]).sum(-2))
    state[slots]=selected
    return torch.stack(outputs,1)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(31051)
    report=dict(passed=False,device='Apple MPS',torch=str(torch.__version__),checks=[],
                source_sha256=hashlib.sha256(Path('damendaemon/hybrid_metal.py').read_bytes()).hexdigest())
    try:
        for length in (1,7,87,185,1024):
            for reset in (True,False):
                slots=torch.tensor([3,1]);shape=(2,length,16,128)
                q,k=[torch.randn(shape) for _ in range(2)]
                q,k=[a*torch.rsqrt(a.square().sum(-1,keepdim=True)+1e-6) for a in (q,k)]
                q=q*(128**-.5)
                v=torch.randn(shape);decay=-torch.rand(shape[:-1]);beta=torch.rand(shape[:-1])
                state=torch.randn(4,16,128,128)*.1
                expected_state=state.clone()
                expected=cpu_scan(q,k,v,decay,beta,expected_state,slots,reset)
                actual_state=state.to('mps')
                actual=scan(*(x.to('mps') for x in (q,k,v,decay,beta)),actual_state,slots.to('mps'),reset).cpu()
                actual_state=actual_state.cpu()
                assert torch.equal(actual_state[[0,2]],state[[0,2]]),'unselected state changed'
                for name,got,target in [('outputs',actual,expected),('state',actual_state,expected_state)]:
                    error=got-target
                    rms=(error.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item()
                    maximum=error.abs().max().item()
                    passed=bool(torch.isfinite(got).all()) and rms<1e-4 and maximum<1e-4
                    report['checks'].append(dict(length=length,reset=reset,kind=name,relative_rms=rms,max_absolute=maximum,passed=passed))
                    assert passed,report['checks'][-1]
        report['passed']=True
    except BaseException as error:report['error']=repr(error);raise
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
