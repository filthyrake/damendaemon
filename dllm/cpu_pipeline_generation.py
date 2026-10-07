"""Owned single-request greedy CPU pipeline generation; token IDs only."""
import threading
import torch
from .cpu_pipeline_build import build_cpu_pipeline,CpuPipelineBuildError
from .hybrid_pipeline import PipelineCancelled


def validate_request(tokens,*,vocab_size,max_context,max_new_tokens,chunk_size,eos_ids=(),excluded_ids=()):
    if (type(vocab_size) is not int or not 1<=vocab_size<=2**21
            or type(max_context) is not int or not 1<=max_context<=2**20
            or type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096
            or type(chunk_size) is not int or not 1<=chunk_size<=4096
            or not isinstance(tokens,(list,tuple)) or not tokens
            or len(tokens)+max_new_tokens>max_context
            or any(type(t) is not int or not 0<=t<vocab_size for t in tokens)):
        raise ValueError('invalid CPU completion tokens/context/output/chunk limits')
    for group in (eos_ids,excluded_ids):
        if (not isinstance(group,(list,tuple,set,frozenset)) or len(group)>256
                or any(type(t) is not int or not 0<=t<vocab_size for t in group)):
            raise ValueError('invalid CPU completion stop/excluded tokens')
    if set(tokens)&set(excluded_ids) or set(eos_ids)&set(excluded_ids) or len(set(excluded_ids))>=vocab_size:
        raise ValueError('CPU completion media/excluded tokens are unsupported')
    return list(tokens)


class CpuPipelineReferenceGenerator:
    def __init__(self,model,budgets,*,vocab_size,eos_ids=(),excluded_ids=(),**stage_options):
        if type(vocab_size) is not int or not 1<=vocab_size<=model.config['vocab_size']:
            raise ValueError('CPU completion tokenizer vocabulary exceeds model')
        media_ids=model.config.get('multimodal_token_ids',[])
        for group in (eos_ids,excluded_ids,media_ids):
            if (not isinstance(group,(list,tuple,set,frozenset)) or len(group)>256
                    or any(type(t) is not int or not 0<=t<vocab_size for t in group)):
                raise ValueError('invalid CPU completion stop/excluded tokens')
        self.model=model;self.budgets=tuple(budgets);self.vocab_size=vocab_size
        self.eos_ids=frozenset(eos_ids);self.excluded_ids=frozenset(excluded_ids)|frozenset(media_ids)
        self.stage_options=dict(stage_options);self.pipeline=None;self.lock=threading.Lock()

    def _close(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline=None

    def cleanup(self):
        if not self.lock.acquire(blocking=False):raise RuntimeError('CPU completion is active')
        try:self._close()
        finally:self.lock.release()

    def generate(self,tokens,*,max_new_tokens=128,chunk_size=128,cancelled=None):
        tokens=validate_request(tokens,vocab_size=self.vocab_size,max_context=self.model.max_context,
                                max_new_tokens=max_new_tokens,chunk_size=chunk_size,
                                eos_ids=self.eos_ids,excluded_ids=self.excluded_ids)
        if cancelled is not None and not callable(cancelled):raise ValueError('cancellation must be callable')
        if not self.lock.acquire(blocking=False):raise RuntimeError('CPU completion is active')
        primary=None;count=0;reason='length'
        try:
            self._close()
            if cancelled is not None and cancelled():raise PipelineCancelled('cancelled before pipeline construction')
            try:self.pipeline=build_cpu_pipeline(self.model,self.budgets,max_batch=1,**self.stage_options)
            except CpuPipelineBuildError as error:
                if not error.closed:self.pipeline=error
                raise
            for start in range(0,len(tokens),chunk_size):
                logits=self.pipeline.prefill_chunk(tokens[start:start+chunk_size],start=start,
                                                   final=start+chunk_size>=len(tokens),cancelled=cancelled)
            for step in range(max_new_tokens):
                if (not isinstance(logits,torch.Tensor) or logits.device.type!='cpu' or logits.dtype!=torch.float32
                        or tuple(logits.shape)!=(1,self.model.config['vocab_size'])
                        or not torch.isfinite(logits[:,:self.vocab_size]).all()):
                    raise ValueError('invalid CPU completion logits')
                scores=logits[0,:self.vocab_size].clone()
                if self.excluded_ids:scores[list(self.excluded_ids)]=-float('inf')
                token=int(scores.argmax());count+=1
                yield dict(token_id=token)
                if token in self.eos_ids:reason='stop';break
                if step+1<max_new_tokens:logits,_=self.pipeline.decode_batch([token],[0],cancelled=cancelled)
        except BaseException as error:primary=error;raise
        finally:
            try:self._close()
            except BaseException as error:
                if primary is None:raise
                primary.add_note(f'CPU pipeline cleanup failed; retain runner and retry cleanup: {type(error).__name__}')
            finally:self.lock.release()
        yield dict(finish_reason=reason,usage=dict(prompt_tokens=len(tokens),completion_tokens=count,
                                                  total_tokens=len(tokens)+count))
