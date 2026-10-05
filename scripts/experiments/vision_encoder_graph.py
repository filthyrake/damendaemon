"""Bounded single-shape CUDA graph experiment; not enabled in serving."""
import torch
from dllm.cuda_warmup import warmup_stream


class GraphVisionEncoder:
    def __init__(self,encoder):
        self.encoder=encoder
        self.key=None
        self.graph=None
        self.pixels=None
        self.output=None

    @torch.inference_mode()
    def encode(self,pixels,grids):
        key=tuple(tuple(grid) for grid in grids)
        if key!=self.key:
            # One cached shape only. Never retain an unbounded set of graph pools.
            self.graph=self.pixels=self.output=None
            self.key=None
            self.pixels=pixels.to(self.encoder.device).clone()
            with warmup_stream():
                for _ in range(3):self.encoder.encode(self.pixels,grids)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.output=self.encoder.encode(self.pixels,grids)
            self.graph=graph
            self.key=key
        self.pixels.copy_(pixels)
        self.graph.replay()
        return self.output
