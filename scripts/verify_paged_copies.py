"""Independent gather/scatter proof with fragmented pages and untouched tails."""
import json
import torch
from dllm.paged_kernels import copy_pages
from dllm.runtime import require_gpu


def main():
    require_gpu()
    torch.manual_seed(927)
    checks=[]
    for dtype in (torch.float16,torch.float32):
        for length in (1,15,16,17,127,513):
            table=torch.full((4,33),-1,device='cuda',dtype=torch.int32)
            pages=torch.randperm(40,device='cuda')[:(length+15)//16]
            table[3,:len(pages)]=pages.to(torch.int32)
            original=torch.randn((2,40,3,16,96),device='cuda',dtype=dtype)
            dense=copy_pages(original,table,3,length)
            expected=torch.cat([original[:,page] for page in pages.tolist()],dim=2)[:,:,:length]
            torch.testing.assert_close(dense,expected,atol=0,rtol=0)
            restored=torch.full_like(original,17)
            copy_pages(restored,table,3,length,snapshot=dense)
            wanted=torch.full_like(original,17)
            for block,page in enumerate(pages.tolist()):
                count=min(16,length-block*16)
                wanted[:,page,:,:count]=expected[:,:,block*16:block*16+count]
            torch.testing.assert_close(restored,wanted,atol=0,rtol=0)
            checks.append({'dtype':str(dtype),'tokens':length,'exact_gather':True,'exact_restore_and_untouched_storage':True})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'checks':checks},indent=2))


if __name__=='__main__':main()
