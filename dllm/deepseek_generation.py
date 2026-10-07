"""Owned, bounded CPU reference text generation; not a serving runtime."""
import codecs
import threading

import torch
from tokenizers import decoders

from .deepseek_text import DeepSeekText, BOS, EOS, USER, ASSISTANT, THINK, END_THINK
from .deepseek_output import DeepSeekOutput
from .floating_model import FloatingModel


def _byte_alphabet():
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    result={chr(value):value for value in visible}
    extra=256
    for value in range(256):
        if value not in visible:
            result[chr(extra)]=value
            extra+=1
    return result


BYTE_VALUES=_byte_alphabet()


class TokenStream:
    """Incremental ByteLevel decoding with explicit control-token handling."""
    def __init__(self,text,*,enable_thinking=False):
        if not isinstance(text,DeepSeekText) or not isinstance(text.tokenizer.decoder,decoders.ByteLevel):
            raise ValueError('V4 generation requires a prepared ByteLevel tokenizer')
        self.text=text
        self.added=text.tokenizer.get_added_tokens_decoder()
        self.utf8=codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.parser=DeepSeekOutput(enable_thinking=enable_thinking)
        self.failed=False

    def feed(self,token):
        if self.failed:raise RuntimeError('V4 token stream failed')
        try:
            if type(token) is not int or not 0 <= token < self.text.vocab_size:
                raise ValueError('invalid generated V4 token ID')
            if token in self.added:
                value=self.added[token].content
                if value not in (BOS,EOS,USER,ASSISTANT,THINK,END_THINK):
                    raise ValueError('unsupported generated V4 added/control token')
                decoded=self.utf8.decode(b'',final=True)+value
                self.utf8.reset()
            else:
                value=self.text.tokenizer.id_to_token(token)
                try:raw=bytes(BYTE_VALUES[char] for char in value)
                except (KeyError,TypeError) as error:raise ValueError('invalid ByteLevel vocabulary token') from error
                decoded=self.utf8.decode(raw,final=False)
            return self.parser.feed(decoded)
        except BaseException:
            self.failed=True
            raise

    def finish(self,*,truncated=False):
        if self.failed:raise RuntimeError('V4 token stream failed')
        try:
            events=self.parser.feed(self.utf8.decode(b'',final=True))
            return events+self.parser.finish(truncated=truncated)
        except BaseException:
            self.failed=True
            raise


def validate_text_config(config,text,max_context):
    """Check declared tokenizer IDs/context before weight loading or prefill."""
    if 'eos_token_id' in config:
        eos=config['eos_token_id']
        eos=eos if isinstance(eos,list) else [eos]
        if any(type(value) is not int for value in eos) or set(eos) != text.eos:
            raise ValueError('model and tokenizer EOS IDs disagree')
    if 'bos_token_id' in config and (type(config['bos_token_id']) is not int
            or config['bos_token_id'] != text.tokenizer.token_to_id(BOS)):
        raise ValueError('model and tokenizer BOS IDs disagree')
    if 'max_position_embeddings' in config and (type(config['max_position_embeddings']) is not int
            or max_context > config['max_position_embeddings']):
        raise ValueError('reference context exceeds model position limit')


class ReferenceGenerator:
    """Greedy CPU reference generation with retryable failed-cleanup ownership.

    Call close() on an abandoned generator iterator. Cache release occurs before
    the final completion event, including EOS and length limits. Failed cleanup
    remains owned by this runner and must succeed before another prefill.
    """
    def __init__(self,model,text,*,max_context=4096,max_cache_bytes=256*1024*1024):
        if (not isinstance(model,FloatingModel) or not isinstance(text,DeepSeekText)
                or model.embedding.device.type != 'cpu' or model.embedding.shape[0] != text.vocab_size
                or not isinstance(text.tokenizer.decoder,decoders.ByteLevel)):
            raise ValueError('reference generator needs matching CPU model and ByteLevel tokenizer')
        if type(max_context) is not int or not 1 <= max_context <= 2**20 or type(max_cache_bytes) is not int or max_cache_bytes < 1:
            raise ValueError('invalid reference context/cache bounds')
        validate_text_config(getattr(model,'config',{}),text,max_context)
        self.model,self.text=model,text
        self.max_context,self.max_cache_bytes=max_context,max_cache_bytes
        self.failed_cleanups=[]
        self.lock=threading.RLock()

    def cleanup(self):
        with self.lock:
            while self.failed_cleanups:
                self.failed_cleanups[0].close()
                self.failed_cleanups.pop(0)

    def _release(self,state):
        try:state.close()
        except BaseException:
            with self.lock:self.failed_cleanups.append(state)
            raise

    def generate(self,messages,*,max_new_tokens=128,enable_thinking=False,drop_thinking=True):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise ValueError('reference output length must be 1..4096 tokens')
        tokens=self.text.chat_tokens(messages,enable_thinking=enable_thinking,drop_thinking=drop_thinking)
        if not tokens or len(tokens)+max_new_tokens > self.max_context:
            raise ValueError('prompt and requested output exceed reference context')
        stream=TokenStream(self.text,enable_thinking=enable_thinking)
        state=None
        primary_error=None
        try:
            with self.lock:
                self.cleanup()
                logits,state=self.model.prefill(torch.tensor([tokens],dtype=torch.int64),
                                                 max_tokens=len(tokens)+max_new_tokens,max_cache_bytes=self.max_cache_bytes)
            reason='length'
            for index in range(max_new_tokens):
                token=int(logits[0].argmax())
                events=stream.feed(token)
                for event in events:yield event
                if token in self.text.eos:
                    reason='stop'
                    break
                if index+1 < max_new_tokens:
                    logits=self.model.decode(torch.tensor([[token]],dtype=torch.int64),state,position=len(tokens)+index)
            for event in stream.finish(truncated=reason == 'length'):yield event
        except BaseException as error:
            primary_error=error
            raise
        finally:
            if state is not None:
                try:self._release(state)
                except BaseException:
                    if primary_error is None:raise
                    primary_error.add_note('reference cache cleanup failed; retained for retry before another prefill')
        yield {'finish_reason':reason,'usage':{'prompt_tokens':len(tokens),'completion_tokens':index+1}}
