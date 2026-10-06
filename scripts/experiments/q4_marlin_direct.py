"""Validation-only preparation using the bitwise-qualified direct permutation."""
import torch

from .q4_marlin import Library, Prepared
from .q4_marlin_repack import repack


class DirectLibrary(Library):
    def prepare(self, weight, scales, bias=None):
        n, k = weight.shape[0], weight.shape[1]*2
        if scales.dtype != torch.float16 or scales.device != weight.device or scales.shape != (n,k//32):
            raise ValueError('invalid Q4 Marlin scales')
        packed = repack(weight,512)
        order = torch.tensor([i+8*j for i in range(8) for j in range(8)],device=weight.device)
        prepared_scales = scales.T.contiguous().reshape(-1,64)[:,order].reshape(k//32,n).contiguous()
        zeros = torch.full((k//32,n//8),-2004318072,dtype=torch.int32,device=weight.device)
        if bias is not None:
            if bias.dtype!=torch.float16 or bias.device!=weight.device or bias.shape!=(n,):
                raise ValueError('invalid Q4 Marlin bias')
            bias_order = torch.tensor([2*i+j for i in range(4) for j in (0,1,8,9,16,17,24,25)],device=weight.device)
            bias = bias.reshape(-1,32)[:,bias_order].reshape(n).contiguous()
        return Prepared(packed,prepared_scales,zeros,bias,k,n)
