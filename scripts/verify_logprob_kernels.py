"""CUDA fused scores against FP64 log-softmax, including ties and masked blocks."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import torch
from dllm.logprob_kernels import score_rows
from dllm.runtime import require_gpu


def main():
    require_gpu();torch.manual_seed(283)
    checks=[]
    for dtype in (torch.float16,torch.float32):
        for vocab in (17,1025,151936,262144):
            logits=torch.randn((4,vocab),device='cuda',dtype=dtype)*3
            chosen=torch.tensor([0,3,vocab-1,vocab//2],device='cuda')
            reference=torch.log_softmax(logits.double(),dim=-1).cpu()
            original=logits.clone()
            for top in (0,5,20):
                packets=score_rows(logits,chosen,[top]*4)
                for row,packet in enumerate(packets):
                    assert packet['token_id']==chosen[row].item()
                    error=abs(packet['logprob']-reference[row,packet['token_id']].item())
                    assert error<.00003,error
                    expected=reference[row].topk(min(top,vocab)).values.tolist()
                    assert len(packet['top'])==len(expected)
                    assert len({token for token,_ in packet['top']})==len(expected)
                    for (token,value),wanted in zip(packet['top'],expected):
                        assert abs(value-wanted)<.00003
                        assert abs(value-reference[row,token].item())<.00003
                assert torch.equal(logits,original)
                checks.append({'dtype':str(dtype),'vocab':vocab,'batch':4,'top':top})
    logits=torch.zeros((1,2049),device='cuda')
    logits[:,1024:2048]=-float('inf')
    chosen=torch.tensor([2048],device='cuda')
    packet=score_rows(logits,chosen,[20])[0]
    assert [token for token,_ in packet['top']]==list(range(20))
    expected=torch.log_softmax(logits.double(),-1)[0,2048].item()
    assert abs(packet['logprob']-expected)<.00003
    for shift in (-1.e6,1.e6):
        logits=torch.randn((4,1025),device='cuda')+shift
        chosen=torch.tensor([0,17,512,1024],device='cuda')
        packets=score_rows(logits,chosen,[5]*4)
        expected=torch.log_softmax(logits.double(),-1)
        for row,packet in enumerate(packets):
            assert abs(packet['logprob']-expected[row,packet['token_id']].item())<.00003
    for value in (float('nan'),float('inf'),-float('inf')):
        invalid=torch.full((1,17),value,device='cuda')
        try:score_rows(invalid,torch.tensor([0],device='cuda'),[5])
        except ValueError:pass
        else:raise AssertionError('invalid distribution was accepted')
    for value in (float('nan'),float('inf')):
        invalid=torch.zeros((1,2049),device='cuda');invalid[0,1300]=value
        try:score_rows(invalid,torch.tensor([0],device='cuda'),[0])
        except ValueError:pass
        else:raise AssertionError('non-finite unselected token was ignored')
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'checks':checks,
        'ties_choose_lowest_id':True,'all_masked_block':True,'large_offsets_stable':True,
        'nonfinite_rejected':True,'logits_unchanged':True},indent=2))


if __name__=='__main__':main()
