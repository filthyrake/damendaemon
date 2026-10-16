"""Opt-in greedy n-gram proposals with full-vocabulary dense CUDA verification."""
import ctypes
import os


def configured_width():
    width=int(os.environ.get('DLLM_SPECULATIVE_TOKENS','0'))
    if width not in (0,2,4,8):raise ValueError('speculative token width must be 0, 2, 4 or 8')
    return width


def eligible(request):
    sampler=request.sampler
    return (not sampler.options.temperature and not sampler.has_adjustments
        and sampler.constraint is None and sampler.logprobs_top is None
        and not request.tools and not request.tool_json and request.schema is None
        and not request.stops and not any(m.get('role')=='tool' or m.get('tool_calls')
            for m in (request.messages or ())))


def propose(history,limit):
    history=history[-1024:]
    for size in (4,3,2,1):
        suffix=history[-size:]
        for start in range(len(history)-size-1,-1,-1):
            if history[start:start+size]==suffix:
                return history[start+size:start+size+limit]
    return []


def create(executor,use_graph):
    width=configured_width()
    family=executor.c['model_type']
    supported=family in ('qwen2','qwen3') or (family=='llama'
        and getattr(executor,'llama_precision',None)=='float32')
    if (not width or not use_graph or executor.device!='cuda' or executor.kernels is None
            or not supported or executor.model.scales or getattr(executor.model,'packed_formats',{})
            or hasattr(executor,'pool') or executor.context<width):
        return None
    return Verifier(executor,width)


class Verifier:
    """One bounded graph per executor; retain no executor reference cycle."""
    def __init__(self,executor,width):
        import torch
        self.width=width;self.context=executor.context;self.slots=executor.max_batch
        self.metadata=torch.zeros((3,width),device='cuda',dtype=torch.long)
        ids,positions,slots=self.metadata.unbind(0)
        positions.copy_(torch.arange(width,device='cuda'))
        self.staging=torch.empty((3,width),dtype=torch.long,pin_memory=True)
        self.staging_values=(ctypes.c_int64*(3*width)).from_address(self.staging.data_ptr())
        self.copied=torch.cuda.Event()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):executor.forward_batch(ids,positions,slots)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.inference_mode(),torch.cuda.graph(self.graph):
            logits=executor.forward_batch(ids,positions,slots)
            from .sampling import greedy_tokens
            self.next_ids=greedy_tokens(logits)

    def verify(self,current,proposals,start,slot):
        import torch
        if (not 1<=len(proposals)<self.width or start<0 or start+self.width>self.context
                or not 0<=slot<self.slots):
            raise ValueError('invalid speculative verification block')
        inputs=[current]+list(proposals)
        self.copied.synchronize()
        self.staging_values[:]=inputs+[0]*(self.width-len(inputs))+list(range(start,start+self.width))+[slot]*self.width
        with torch.inference_mode():
            self.metadata.copy_(self.staging,non_blocking=True);self.copied.record()
            self.graph.replay();predictions=self.next_ids.tolist()
        if any(token<0 for token in predictions[:len(inputs)]):
            raise RuntimeError('non-finite speculative verification logits')
        accepted=0
        for actual,draft in zip(predictions,proposals):
            if actual!=draft:break
            accepted+=1
        return list(proposals[:accepted])+[predictions[accepted]],accepted
