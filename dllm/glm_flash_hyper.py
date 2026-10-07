"""Bounded CPU FP32 GLM-5.3-Flash residual-stream reference primitives.

Independent transcription of Transformers v5.17.0 glm5_next equations. Shared
Sinkhorn mixing is also used by DeepSeek, but Flash normalizes before projecting
and its final head takes an unweighted mean. Parameters are borrowed, not copied.
No model admission or hardware qualification is implied.
"""
import torch
from torch.nn import functional as F
from .hyper_connections import mixing_weights,restore_streams


def _tensor(value,shape):
    return (isinstance(value,torch.Tensor) and value.device.type=='cpu'
            and value.dtype==torch.float32 and tuple(value.shape)==tuple(shape))


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('nonfinite Flash hyper-connection value')


class FlashHyperConnection:
    def __init__(self,projection,base,scale,*,streams=4,norm_epsilon=1e-5,
                 connection_epsilon=1e-6,iterations=20,max_workspace_bytes=64*1024**2,
                 max_output_bytes=64*1024**2):
        if (type(streams) is not int or not 1<=streams<=16
                or not isinstance(projection,torch.Tensor) or projection.ndim!=2
                or projection.shape[1]%streams or not 1<=projection.shape[1]//streams<=2**21
                or not _tensor(projection,(streams*(streams+2),projection.shape[1]))
                or not _tensor(base,(streams*(streams+2),)) or not _tensor(scale,(3,))
                or type(iterations) is not int or not 1<=iterations<=128
                or any(type(v) not in (int,float) or not 0<v<=1 for v in (norm_epsilon,connection_epsilon))
                or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
            raise ValueError('invalid Flash hyper-connection parameters or limits')
        _finite(projection,base,scale)
        self.projection,self.base,self.scale=projection,base,scale
        self.streams,self.width=streams,projection.shape[1]//streams
        self.norm_epsilon,self.connection_epsilon,self.iterations=norm_epsilon,connection_epsilon,iterations
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes

    def _admit(self,hidden,operation):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=4
                or not _tensor(hidden,(*hidden.shape[:2],self.streams,self.width))
                or min(hidden.shape[:2])<1):
            raise ValueError('Flash residual state requires nonempty CPU FP32 [batch, sequence, streams, width]')
        rows=hidden.shape[0]*hidden.shape[1];h=self.streams;d=self.width
        # Conservative live tensor allowance, including validation masks and
        # noncontiguous input copies; excludes borrowed parameters/input storage.
        workspace=64*rows*(h*d+h*(h+2)+h*h+d)
        output=4*rows*({'reduce':d+h+h*h,'restore':h*d,'head':d}[operation])
        if workspace>self.max_workspace_bytes or output>self.max_output_bytes:
            raise ValueError('Flash hyper-connection workspace/output budget exceeded')
        return hidden.shape[:2]

    @torch.no_grad()
    def reduce(self,hidden):
        self._admit(hidden,'reduce');_finite(hidden,self.projection,self.base,self.scale)
        flat=hidden.flatten(-2)
        variance=flat.square().mean(-1,keepdim=True)
        _finite(variance) # do not silently turn finite-but-overflowing state into zeros
        normalized=flat*torch.rsqrt(variance+self.norm_epsilon)
        mixes=F.linear(normalized,self.projection)
        _finite(mixes)
        pre,post,matrix=mixing_weights(mixes,self.scale,self.base,self.streams,
                                      iterations=self.iterations,epsilon=self.connection_epsilon)
        reduced=(pre.unsqueeze(-1)*hidden).sum(-2)
        _finite(reduced,post,matrix)
        return reduced,post,matrix

    @torch.no_grad()
    def restore(self,output,residual,post,matrix):
        b,s=self._admit(residual,'restore');h=self.streams
        if not _tensor(output,(b,s,self.width)) or not _tensor(post,(b,s,h)) or not _tensor(matrix,(b,s,h,h)):
            raise ValueError('invalid Flash residual restoration geometry')
        _finite(output,residual,post,matrix)
        result=restore_streams(output,residual,post,matrix)
        _finite(result)
        return result

    @torch.no_grad()
    def head(self,hidden):
        self._admit(hidden,'head');_finite(hidden)
        result=hidden.mean(-2)
        _finite(result)
        return result
