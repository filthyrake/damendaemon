"""Owned bounded greedy GLM CPU reference generation, separate from serving."""
import threading
import torch
from tokenizers import decoders
from .bytelevel_stream import ByteLevelStream
from .glm_text import GlmText,CONTROLS,EOS
from .glm_output import GlmOutput
from .glm_model import GlmModel


class TokenStream:
    def __init__(self,text):
        if not isinstance(text,GlmText):raise ValueError('GLM output requires a prepared tokenizer')
        self.text=text;self.decoder=ByteLevelStream(text.tokenizer,CONTROLS)
        self.parser=GlmOutput();self.failed=False

    def feed(self,token):
        if self.failed:raise RuntimeError('GLM token stream failed')
        try:
            if type(token) is not int or not 0<=token<self.text.tokenizer_size:
                raise ValueError('invalid or unmapped generated GLM token ID')
            return self.parser.feed(self.decoder.feed(token))
        except BaseException:self.failed=True;raise

    def finish(self,*,truncated=False):
        if self.failed:raise RuntimeError('GLM token stream failed')
        try:return self.parser.feed(self.decoder.finish())+self.parser.finish(truncated=truncated)
        except BaseException:self.failed=True;raise


def validate_text_config(config,text,max_context):
    if not isinstance(config,dict):raise ValueError('GLM model configuration must be a mapping')
    if 'eos_token_id' in config:
        eos=config['eos_token_id']
        if not isinstance(eos,list) or any(type(v) is not int for v in eos) or set(eos)!=text.eos:
            raise ValueError('GLM model and tokenizer stop IDs disagree')
    if config.get('bos_token_id') is not None:raise ValueError('GLM template does not use a model BOS token')
    if 'pad_token_id' in config and (type(config['pad_token_id']) is not int or config['pad_token_id']!=text.tokenizer.token_to_id(EOS)):
        raise ValueError('GLM model and tokenizer pad IDs disagree')
    limit=config.get('max_position_embeddings',2**20)
    if type(limit) is not int or not 1<=max_context<=limit:
        raise ValueError('GLM reference context exceeds model position limit')


class ReferenceGenerator:
    """Close abandoned iterators; failed releases stay owned until cleanup succeeds."""
    def __init__(self,model,text,*,max_context=4096,max_cache_bytes=256*1024**2):
        if (not isinstance(model,GlmModel) or not isinstance(text,GlmText)
                or model.embedding.device.type!='cpu' or model.embedding.shape[0]!=text.vocab_size
                or not isinstance(text.tokenizer.decoder,decoders.ByteLevel)):
            raise ValueError('GLM generation requires matching CPU model and ByteLevel tokenizer')
        if (type(max_context) is not int or not 1<=max_context<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31):
            raise ValueError('invalid GLM generation context/cache budget')
        validate_text_config(getattr(model,'config',{}),text,max_context)
        self.model,self.text=model,text;self.max_context,self.max_cache_bytes=max_context,max_cache_bytes
        self.failed_cleanups=[];self.lock=threading.RLock()

    def cleanup(self):
        with self.lock:
            while self.failed_cleanups:
                self.failed_cleanups[0].close();self.failed_cleanups.pop(0)
            self.model.cleanup()

    def _release(self,state):
        try:state.close()
        except BaseException:
            with self.lock:self.failed_cleanups.append(state)
            raise

    def generate(self,messages,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:
            raise ValueError('GLM output length must be 1..4096 tokens')
        tokens=self.text.chat_tokens(messages,reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        if not tokens or len(tokens)+max_new_tokens>self.max_context:
            raise ValueError('GLM prompt and requested output exceed context')
        stream=TokenStream(self.text);state=None;primary=None
        try:
            with self.lock:
                self.cleanup()
                logits,state=self.model.prefill(torch.tensor([tokens],dtype=torch.int64,device='cpu'),
                                                 capacity=len(tokens)+max_new_tokens,max_cache_bytes=self.max_cache_bytes)
            reason='length'
            for index in range(max_new_tokens):
                if (not isinstance(logits,torch.Tensor) or logits.shape!=(1,self.text.vocab_size)
                        or logits.device.type!='cpu' or logits.dtype!=torch.float32):
                    raise ValueError('invalid GLM generation logits')
                # Unmapped padded rows are excluded without mutating model logits.
                mapped=logits[0,:self.text.tokenizer_size]
                if not torch.isfinite(mapped).all():raise ValueError('nonfinite GLM generation logits')
                token=int(mapped.argmax())
                for event in stream.feed(token):yield event
                if token in self.text.eos:reason='stop';break
                if index+1<max_new_tokens:
                    logits=self.model.forward(torch.tensor([[token]],dtype=torch.int64,device='cpu'),state,position=len(tokens)+index)
            yield from stream.finish(truncated=reason=='length')
        except BaseException as error:primary=error;raise
        finally:
            if state is not None:
                try:self._release(state)
                except BaseException:
                    if primary is None:raise
                    primary.add_note('GLM cache cleanup failed; retained for retry before another prefill')
        yield {'finish_reason':reason,'usage':{'prompt_tokens':len(tokens),'completion_tokens':index+1}}
