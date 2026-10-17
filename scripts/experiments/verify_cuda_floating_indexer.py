"""Tiny dense/packed CUDA learned indexer composition against CPU scalar scores."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_floating_indexer import index_hidden
    from damendaemon.cuda_prepared_projection import CudaPackedProjection
    from damendaemon.activation_simulation import normalized_hadamard,simulate_fp4
    from tests.test_floating_indexer import FloatingIndexerTests
    from tests.test_compressed_indexer import oracle
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        host=FloatingIndexerTests().fixture(torch.float32)
        for packed in (False,True):
            h,lr,qw,hw,keys,pos,freq=host
            def prepare(w):
                if not packed:return w.cuda(),w
                raw=w.to(torch.float8_e4m3fn).view(torch.uint8)
                scales=torch.full(((w.shape[0]+127)//128,(w.shape[1]+127)//128),127,dtype=torch.uint8)
                return CudaPackedProjection(raw.cuda(),scales.cuda(),format='e4m3fn',row_chunk=16,column_chunk=16,workspace_bytes=65536),raw.view(torch.float8_e4m3fn).float()
            qgpu,qcpu=prepare(qw);hgpu,hcpu=prepare(hw)
            query=(lr@qcpu.T).reshape(2,19,3,32)
            for b in range(2):
                for t in range(19):
                    for head in range(3):
                        for pair in range(8):
                            d=16+2*pair;a,c=query[b,t,head,d:d+2].tolist();angle=float(pos[b,t])*float(freq[pair])
                            query[b,t,head,d]=a*math.cos(angle)-c*math.sin(angle)
                            query[b,t,head,d+1]=c*math.cos(angle)+a*math.sin(angle)
            query=simulate_fp4(normalized_hadamard(query))
            expected=oracle(query,keys,h@hcpu.T,pos,4,3,128)
            device=[h.cuda(),lr.cuda(),qgpu,hgpu,keys.cuda(),pos.cuda(),freq.cuda()]
            for chunk in (1,3,16):
                ids,scores=index_hidden(*device,top_k=3,offset=128,query_chunk=chunk,key_chunk=2)
                torch.testing.assert_close(ids.cpu(),expected[0],atol=0,rtol=0)
                torch.testing.assert_close(scores.cpu(),expected[1],atol=2e-5,rtol=3e-5)
                mask=torch.isfinite(expected[1]);error=float((scores.cpu()[mask]-expected[1][mask]).abs().max())
                report['cases'].append(dict(packed=packed,query_chunk=chunk,ids_exact=True,max_absolute=error))
            for option in ('max_workspace_bytes','max_output_bytes'):
                with patch('damendaemon.cuda_floating_indexer._linear',side_effect=AssertionError('projection before admission')):
                    rejected(lambda:index_hidden(*device,top_k=3,**{option:1}))
            bad=list(device);bad[1]=lr.cpu();rejected(lambda:index_hidden(*bad,top_k=3))
            bad=list(device);bad[6]=device[6]+float('nan');rejected(lambda:index_hidden(*bad,top_k=3))
            bad=list(device);bad[5]=device[5]-100;rejected(lambda:index_hidden(*bad,top_k=3))
            bad=list(device);bad[1]=device[1]*0+1e30;bad[2]=torch.full(qw.shape,1e30,device='cuda');rejected(lambda:index_hidden(*bad,top_k=3))
        report.update(passed=True,preallocation_bounds=True,invalid_intermediates_rejected=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
