"""Diagnostic replay of mixed-precision Llama residual arithmetic (no graphs)."""
import sys
import torch
import replay_reference

original=replay_reference.make_executor

def candidate(*args,**kwargs):
    kwargs['use_graph']=False
    executor=original(*args,**kwargs)
    if executor.c['model_type']!='llama':
        raise ValueError('this numerical probe is Llama-only')
    executor.residual_dtype=torch.float32
    executor.precise=True
    executor.rotary_precise=True
    return executor


if __name__=='__main__':
    replay_reference.make_executor=candidate
    if '--no-graph' not in sys.argv:sys.argv.append('--no-graph')
    replay_reference.main()
