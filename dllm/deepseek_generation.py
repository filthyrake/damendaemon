"""Owned, bounded CPU reference text generation; not a serving runtime."""
import threading

import torch
from tokenizers import decoders

from .deepseek_text import DeepSeekText, BOS, EOS, USER, ASSISTANT, THINK, END_THINK
from .deepseek_output import DeepSeekOutput
from .floating_model import FloatingModel


from .bytelevel_stream import BYTE_VALUES,ByteLevelStream


class TokenStream:
    """Incremental ByteLevel decoding with explicit control-token handling."""
    def __init__(self,text,*,enable_thinking=False,allow_tool_calls=False):
        if not isinstance(text,DeepSeekText) or not isinstance(text.tokenizer.decoder,decoders.ByteLevel):
            raise ValueError('V4 generation requires a prepared ByteLevel tokenizer')
        self.text=text
        allowed=(BOS,EOS,USER,ASSISTANT,THINK,END_THINK)+(('｜DSML｜',) if allow_tool_calls else ())
        self.decoder=ByteLevelStream(text.tokenizer,allowed)
        self.parser=DeepSeekOutput(enable_thinking=enable_thinking,allow_tool_calls=allow_tool_calls)
        self.allow_tool_calls=allow_tool_calls
        self.failed=False

    def feed(self,token):
        if self.failed:raise RuntimeError('V4 token stream failed')
        try:
            if type(token) is not int or not 0 <= token < self.text.vocab_size:
                raise ValueError('invalid generated V4 token ID')
            decoded=self.decoder.feed(token)
            return self.parser.feed(decoded)
        except BaseException:
            self.failed=True
            raise

    def finish(self,*,truncated=False):
        if self.failed:raise RuntimeError('V4 token stream failed')
        try:
            events=self.parser.feed(self.decoder.finish())
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

    def generate(self,messages,*,max_new_tokens=128,enable_thinking=False,drop_thinking=True,tools=None):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise ValueError('reference output length must be 1..4096 tokens')
        tokens=self.text.chat_tokens(messages,enable_thinking=enable_thinking,drop_thinking=drop_thinking,tools=tools)
        if not tokens or len(tokens)+max_new_tokens > self.max_context:
            raise ValueError('prompt and requested output exceed reference context')
        stream=TokenStream(self.text,enable_thinking=enable_thinking,allow_tool_calls=tools is not None)
        state=None
        primary_error=None
        try:
            with self.lock:
                self.cleanup()
                logits,state=self.model.prefill(torch.tensor([tokens],dtype=torch.int64),
                                                 max_tokens=len(tokens)+max_new_tokens,max_cache_bytes=self.max_cache_bytes)
            reason='length'
            for index in range(max_new_tokens):
                if (not isinstance(logits,torch.Tensor) or logits.shape != (1,self.text.vocab_size)
                        or logits.device.type != 'cpu' or logits.dtype != torch.float32):
                    raise ValueError('invalid V4 generation logits')
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite V4 generation logits')
                token=int(logits[0].argmax())
                events=stream.feed(token)
                for event in events:yield event
                if token in self.text.eos:
                    reason='stop'
                    break
                if index+1 < max_new_tokens:
                    logits=self.model.decode(torch.tensor([[token]],dtype=torch.int64),state,position=len(tokens)+index)
            for event in stream.finish(truncated=reason == 'length'):
                if event.get('tool_calls'):
                    names={tool['function']['name'] for tool in tools}
                    if any(call['function']['name'] not in names for call in event['tool_calls']):
                        raise ValueError('generated undeclared V4 tool')
                    reason='tool_calls'
                yield event
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
