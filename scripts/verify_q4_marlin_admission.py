"""Real-device Q4 Marlin pre-transfer admission and executor release gate."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import weakref
from unittest.mock import patch

import torch

from dllm.gguf_model import GgufWeights
from dllm.qwen2 import Executor
from dllm.runtime import require_gpu


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    path=Path(os.environ['DLLM_Q4_MARLIN_LIBRARY']).resolve()
    source=GgufWeights(args.model,512,'none')
    report=dict(passed=False,model=args.model,device=torch.cuda.get_device_name(),
                library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cycles=[])
    try:
        # No tensor transfer is allowed while determining the real required sum.
        rejected=Executor.__new__(Executor)
        original_to=torch.Tensor.to
        def forbid_cuda(tensor,*args,**kwargs):
            device=kwargs.get('device',args[0] if args else None)
            if isinstance(device,(str,torch.device)) and torch.device(device).type=='cuda':
                raise AssertionError('GPU transfer happened before admission')
            return original_to(tensor,*args,**kwargs)
        torch.cuda.empty_cache()
        before=torch.cuda.memory_allocated()
        for budget in (0,None):
            if budget is None:budget=rejected.required_memory_bytes-1
            with patch('dllm.qwen2.available_memory',return_value=budget), \
                 patch.object(torch.Tensor,'to',forbid_cuda), \
                 patch('dllm.native_q4_marlin.Library.allocate_workspace',side_effect=AssertionError('workspace before admission')):
                try:Executor.__init__(rejected,source,use_graph=False,max_batch=4)
                except MemoryError:pass
                else:raise AssertionError('insufficient memory admitted')
            assert torch.cuda.memory_allocated()==before
        report['required_bytes']=rejected.required_memory_bytes
        report['layout_overhead_bytes']=rejected.packed_layout_overhead_bytes
        assert rejected.packed_layout_overhead_bytes==111099904
        report['workspace_reserve_bytes']=rejected.packed_workspace_bytes
        assert rejected.packed_workspace_bytes==81920512+2621600
        del rejected
        prompt=source.raw_tokens('Explain binary search in simple terms.')
        expected=None
        for cycle in range(2):
            with torch.inference_mode():
                executor=Executor(source,use_graph=True,max_batch=4)
                assert '+cuda-q4-marlin' in executor.kernel_backend
                linear=executor.quant_linear
                assert len(linear.q4_marlin)==129
                refs=[weakref.ref(executor),weakref.ref(linear),weakref.ref(linear.native_q4_marlin)]
                refs += [weakref.ref(t) for matrix in linear.q4_marlin.values()
                         for t in (matrix.weight,matrix.scales,matrix.zeros)]
                refs += [weakref.ref(linear.native_q4_marlin.scratch),weakref.ref(linear.native_q4_marlin.locks)]
                logits=executor.prefill(prompt)
                tokens=[]
                for _ in range(8):
                    token=logits.argmax(-1).item()
                    tokens.append(token)
                    logits,next_tokens=executor.decode(token)
                    del next_tokens
                if expected is None:expected=tokens
                assert tokens==expected
                allocated=torch.cuda.memory_allocated()
                del logits,executor,linear
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            assert all(ref() is None for ref in refs),'executor GPU state leaked'
            assert torch.cuda.memory_allocated()==before,'GPU allocation not released'
            report['cycles'].append(dict(tokens=tokens,allocated_bytes=allocated,released_bytes=allocated-before))
        report['passed']=True
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
