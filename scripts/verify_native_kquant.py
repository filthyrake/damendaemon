"""Independent coefficient probes for the optional CUDA C++ K-quant decoder."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from dllm.native_kquant import linear_if_enabled
from verify_kquant_kernels import fixture


def main():
    path=os.environ['DLLM_KQUANT_CUDA_LIBRARY']
    torch.set_num_threads(4)
    rng=np.random.default_rng(913);reports=[]
    with torch.inference_mode():
        for width in (256,768,2048,8192):
            for kinds in ((12,),(13,),(14,),(14,12,13)):
                parts=[];values=[];spec=[];first=0;offset=0
                for i,kind in enumerate(kinds):
                    rows=17+i
                    data,expected=fixture(rng,kind,rows,width)
                    parts.append(data);values.append(expected)
                    spec.append((first,first+rows,kind,offset));first+=rows;offset+=data.nbytes
                weight=torch.from_numpy(np.concatenate(parts)).cuda()
                reference=torch.from_numpy(np.concatenate(values)).cuda().half()
                layout=(width,tuple(spec))
                # Exhaust the smaller widths and every coefficient in the
                # first two and final blocks of wider matrices. Mixed layouts
                # include a Q4_K segment starting at a two-byte-aligned offset.
                columns=list(range(width)) if width<=768 else list(range(512))+list(range(width-256,width))
                for start in range(0,len(columns),4):
                    selected=columns[start:start+4]
                    x=torch.zeros((len(selected),width),device='cuda',dtype=torch.float16)
                    x[torch.arange(len(selected),device='cuda'),selected]=1
                    actual=linear_if_enabled(x,weight,layout)
                    assert torch.equal(actual,reference[:,selected].T),(width,kinds,selected)
                reports.append(dict(width=width,kinds=kinds,rows=first,columns=len(columns),bitwise_coefficients=True))
    with Path(path).open('rb') as file:sha=hashlib.file_digest(file,'sha256').hexdigest()
    print(json.dumps(dict(passed=True,device=torch.cuda.get_device_name(),native_library=path,
        native_library_sha256=sha,cases=reports),indent=2))


if __name__=='__main__':main()
