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
    lengths=[]
    report=dict(passed=False,device='Apple MPS',torch=str(torch.__version__),checks=[],
                geometries=[dict(hidden_size=1024,linear_num_key_heads=16,
                                 linear_key_head_dim=128,linear_value_head_dim=128,
                                 note='Qwen3.5-0.8B text scan geometry'),
                            dict(hidden_size=2048,linear_num_key_heads=16,
                                 linear_key_head_dim=128,linear_value_head_dim=128,
                                 note='Qwen3.5-2B text scan geometry — identical '
                                      'scan shapes; hidden_size is not a scan input')],
                lengths=lengths,source_sha256=hashlib.sha256(Path('damendaemon/hybrid_metal.py').read_bytes()).hexdigest())
    try:
        # #4: lengths extended past the old 1024 bound to cover the widened
        # eligibility context cap. 32K covers a whole-prompt unchunked scan
        # at the maximum admitted context (chunked prefill bounds per-call
        # length to the chunk, but the kernel contract is verified at the
        # full bound regardless).
        for length in (1,7,87,185,1024,2048,4096,16384,32768):
            report['lengths'].append(length)
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
