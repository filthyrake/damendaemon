"""Tiny bounded CUDA selected-KV/sink gate against independent scalar softmax."""
import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_selected_attention import selected_attention
    from tests.test_selected_attention import SelectedAttentionTests,scalar_attention
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
        host=SelectedAttentionTests().fixture();device=[v.cuda() for v in host];originals=[v.clone() for v in device]
        expected=scalar_attention(*host,scale=.3)
        for qc,kc in ((1,1),(2,3),(16,64)):
            actual=selected_attention(*device,scale=.3,query_chunk=qc,key_chunk=kc)
            torch.testing.assert_close(actual.cpu(),expected,atol=2e-6,rtol=2e-5)
            report['cases'].append(dict(query_chunk=qc,key_chunk=kc,max_absolute=float((actual.cpu()-expected).abs().max())))
        q,kv,ids,sink=device
        poison=torch.full_like(kv,float('nan'));padding=torch.full_like(ids,-1)
        torch.testing.assert_close(selected_attention(q,poison,padding,sink),torch.zeros_like(q),atol=0,rtol=0)
        torch.testing.assert_close(selected_attention(q,poison,ids[...,:0],sink),torch.zeros_like(q),atol=0,rtol=0)
        assert selected_attention(q[:,:0],kv,ids[:,:0],sink).shape==(2,0,3,4)
        rejected(lambda:selected_attention(q,poison,ids,sink))
        # Invalid positive IDs must never reach CUDA's gather (which can poison a context).
        with patch('dllm.cuda_selected_attention._selected_attention',side_effect=AssertionError('gather entered')):
            rejected(lambda:selected_attention(q,kv,torch.full_like(ids,kv.shape[1]),sink))
            for option in ('max_workspace_bytes','max_output_bytes'):
                rejected(lambda:selected_attention(*device,**{option:1}))
        rejected(lambda:selected_attention(q,kv,ids.cpu(),sink))
        rejected(lambda:selected_attention(q*0+1e30,kv*0+1e30,ids,sink))
        rejected(lambda:selected_attention(q,kv,ids,sink+float('nan')))
        large=[torch.tensor([[[[1000.,0.],[-1000.,0.]]]]),torch.tensor([[[1.,2.],[-1.,3.]]]),torch.tensor([[[0,1,0]]]),torch.tensor([1001.,-1001.])]
        actual=selected_attention(*(v.cuda() for v in large),scale=1.,key_chunk=1)
        torch.testing.assert_close(actual.cpu(),scalar_attention(*large,scale=1.),atol=2e-6,rtol=2e-5)
        for a,b in zip(device,originals):torch.testing.assert_close(a,b,atol=0,rtol=0)
        report.update(passed=True,padding_poison_masked=True,indices_checked_before_gather=True,overflow_rejected=True,inputs_unchanged=True,large_scores_finite=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
