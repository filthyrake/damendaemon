"""Tiny deterministic prepared CUDA/decoded CPU blocks for guarded checks."""
import torch
from dllm.floating_block import FloatingBlock
from dllm.cuda_deepseek_block import CudaDeepSeekBlock
from dllm.cuda_prepared_projection import CudaPackedProjection
from dllm.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
from tests.test_floating_block import fixture
from tests.test_packed_attention import NAMES


def prepare_block(ratio,hashed):
    def raw(value):
        data=value.to(torch.float8_e4m3fn).view(torch.uint8)
        scales=torch.full(((value.shape[0]+127)//128,(value.shape[1]+127)//128),127,dtype=torch.uint8)
        return data,scales,data.view(torch.float8_e4m3fn).float()
    block,hidden,ids=fixture(hashed,ratio)
    attention=dict(block.attention_weights);gpu_attention={}
    for name,value in attention.items():
        if name in NAMES:
            data,scales,attention[name]=raw(value)
            gpu_attention[name]=CudaPackedProjection(data.cuda(),scales.cuda(),format='e4m3fn',row_chunk=16,column_chunk=16,workspace_bytes=65536)
        else:gpu_attention[name]=value.cuda()
    weights=dict(block.weights);bank=[]
    for index in range(4):
        matrices=[]
        for part in ('w1','w3','w2'):
            name=('ffn.experts.' if index<3 else 'ffn.shared_experts.')+part+'.weight'
            value=weights[name][index] if index<3 else weights[name]
            data,scales,decoded=raw(value)
            if index<3:
                if index==0:weights[name]=weights[name].clone()
                weights[name][index]=decoded
            else:weights[name]=decoded
            matrices.append(CudaPackedMatrix(data.cuda(),scales.cuda(),'e4m3fn'))
        bank.append(CudaPackedExpert(*matrices,row_chunk=16,column_chunk=16,workspace_bytes=65536,activation_limit=10.))
    options=dict(groups=2,window=4,ratio=ratio,hc_mult=2,top_k=2,index_top_k=2)
    control=FloatingBlock(attention,weights,block.frequencies,**options)
    gpu_weights={k:v.cuda() for k,v in weights.items() if not k.startswith(('ffn.experts.','ffn.shared_experts.'))}
    candidate=CudaDeepSeekBlock(gpu_attention,gpu_weights,block.frequencies.cuda(),packed_experts=bank[:3],packed_shared=bank[3],**options)
    return candidate,control,hidden,ids
