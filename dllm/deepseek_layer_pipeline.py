"""Complete DeepSeek CPU pipelines preserving token-dependent routing."""
import torch
from .deepseek_layer_stage import DeepSeekLayerStage
from .layer_pipeline import CpuLayerPipeline,LayerPipelineState as DeepSeekPipelineState


class DeepSeekLayerPipeline(CpuLayerPipeline):
    stage_type=DeepSeekLayerStage
    cleanup_failed_prefills=CpuLayerPipeline.cleanup

    def __init__(self,stages):
        super().__init__(stages);self.streams=self.stages[0].streams
        if any(stage.streams!=self.streams or stage.vocab_size!=self.embedding.shape[0] for stage in self.stages):
            raise ValueError('DeepSeek pipeline stream/vocabulary geometry disagrees')

    def _head_shape(self,stage):return stage.head['head.weight'].shape

    @property
    def boundary_width(self):return self.streams*self.stages[0].dim

    @property
    def cleanup_pending(self):return bool(self.failed_cleanups or any(stage.prefill_cleanup_pending for stage in self.stages))

    @property
    def prefill_cleanup_pending(self):return self.cleanup_pending

    def _stage_batch(self,state):return state.layers[0].attention.cache.shape[0]

    def cache_bytes_per_row(self,capacity):return sum(stage.cache_bytes_per_slot(capacity) for stage in self.stages)

    def _tokens(self,tokens,all_logits=False):
        if type(all_logits) is not bool or all_logits:raise ValueError('DeepSeek pipeline returns only final-token logits')
        self.stages[0]._input(tokens,None,False)
        self._input_budget(tokens.shape,False)

    def new_state(self,*args,**options):raise ValueError('DeepSeek pipeline cache construction requires token-bearing prefill')

    @torch.no_grad()
    def prefill(self,tokens,*,max_tokens=None,max_cache_bytes=256*1024**2,query_chunk=16):
        self._tokens(tokens);length=tokens.shape[1];capacity=length if max_tokens is None else max_tokens
        if (type(capacity) is not int or not length<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31
                or type(query_chunk) is not int or not 1<=query_chunk<=128
                or tokens.shape[0]*self.cache_bytes_per_row(capacity)>max_cache_bytes):
            raise ValueError('invalid DeepSeek pipeline prefill controls or aggregate cache budget')
        with self.lock:
            self.cleanup();state=DeepSeekPipelineState(self,[],capacity,max_cache_bytes,tokens.shape[0],position=length)
            hidden=None
            try:
                for stage in self.stages:
                    hidden,local=stage.prefill(tokens,hidden=hidden,max_tokens=capacity,max_cache_bytes=max_cache_bytes,query_chunk=query_chunk)
                    state.stages.append(local)
            except BaseException as error:self._release_failed(state,error);raise
            return hidden,state

    @torch.no_grad()
    def decode(self,tokens,state,*,position):
        self._tokens(tokens)
        if tokens.shape[1]!=1:raise ValueError('DeepSeek pipeline decode needs one token per row')
        if not isinstance(state,DeepSeekPipelineState) or state.owner is not self:raise ValueError('DeepSeek pipeline cache owner differs')
        with state.lock:
            state._usable()
            if type(position) is not int or position!=state.position or position>=state.capacity or tokens.shape[0]!=state.batch:
                raise ValueError('DeepSeek pipeline cache cursor/shape differs')
            hidden=None
            try:
                for stage,local in zip(self.stages,state.stages):hidden=stage.decode(tokens,local,position=position,hidden=hidden)
            except BaseException:state._poison();raise
            state.position=position+1
            return hidden

    def forward(self,tokens,state,*,position,all_logits=False):
        self._tokens(tokens,all_logits)
        return self.decode(tokens,state,position=position)
