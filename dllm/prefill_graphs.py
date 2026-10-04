"""Bounded, opt-in CUDA graph reuse for qualified single-prompt prefills."""
from collections import OrderedDict
import os


def configuration():
    count=int(os.environ.get('DLLM_PREFILL_GRAPHS','0'))
    tokens=int(os.environ.get('DLLM_PREFILL_GRAPH_MAX_TOKENS','128'))
    if not 0<=count<=8 or not 1<=tokens<=512:
        raise ValueError('prefill graphs require 0..8 entries and a 1..512 token limit')
    return count,tokens


class PrefillGraphs:
    def __init__(self,limit,max_tokens):
        import torch
        self.limit=limit;self.max_tokens=max_tokens
        self.entries=OrderedDict();self.hits=0;self.captures=0;self.evictions=0
        self.stream=torch.cuda.Stream()

    def run(self,ids,slot,forward,head_key=None,head=None):
        import torch
        # Candidate identity is part of the graph key. Keep its immutable
        # selected head alive for every replay; full-head graphs use None.
        key=(ids.numel(),slot,head_key)
        entry=self.entries.get(key)
        if entry is not None:
            staged,_,graph,logits,_=entry
            staged.copy_(ids);graph.replay();self.entries.move_to_end(key);self.hits+=1
            return logits
        # Evict before allocating a new private graph pool. No closures holding
        # the executor or graph are retained by this cache.
        while len(self.entries)>=self.limit:
            self.entries.popitem(last=False);self.evictions+=1
        staged=ids.clone();positions=torch.arange(ids.numel(),device=ids.device)
        stream=self.stream;stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):forward(staged,positions,prefill=True,slot=slot)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):logits=forward(staged,positions,prefill=True,slot=slot)
        graph.replay()
        self.entries[key]=(staged,positions,graph,logits,head);self.captures+=1
        return logits
