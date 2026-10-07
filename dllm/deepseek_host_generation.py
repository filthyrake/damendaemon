"""Pinned native DeepSeek chat/DSML generation over owned CPU host stages."""
from contextlib import closing
import hashlib
import json
from tokenizers import Tokenizer,decoders
from .deepseek_text import DeepSeekText,BOS,EOS,USER,ASSISTANT,THINK,END_THINK
from .deepseek_generation import TokenStream,validate_text_config
from .deepseek_tool_prompt import prepare_tools,validate_tool_arguments
from .native_host_generation import NativeHostGenerator
from .reference_batch import prepare_batch


def _snapshot(text):
    if type(text) is not DeepSeekText or type(text.vocab_size) is not int or not 1<=text.vocab_size<=2**21:
        raise ValueError('DeepSeek host generation requires its exact native tokenizer')
    encoded=text.tokenizer.to_str()
    if len(encoded.encode())>64*1024**2:raise ValueError('DeepSeek host tokenizer exceeds metadata bounds')
    tokenizer=Tokenizer.from_str(encoded);tokenizer.no_padding();tokenizer.no_truncation()
    vocab=tokenizer.get_vocab()
    if (not isinstance(tokenizer.decoder,decoders.ByteLevel) or len(vocab)!=text.vocab_size
            or set(vocab.values())!=set(range(text.vocab_size))
            or any(tokenizer.token_to_id(marker) is None or tokenizer.encode(marker,add_special_tokens=False).ids!=[tokenizer.token_to_id(marker)]
                   for marker in (BOS,EOS,USER,ASSISTANT,THINK,END_THINK))):
        raise ValueError('invalid DeepSeek host tokenizer/control mapping')
    eos=frozenset([tokenizer.token_to_id(EOS)])
    if text.eos!=eos:raise ValueError('DeepSeek host stop policy disagrees with EOS')
    result=DeepSeekText.__new__(DeepSeekText)
    result.tokenizer,result.vocab_size,result.tokenizer_size,result.eos=tokenizer,text.vocab_size,len(vocab),eos
    return result


def _policy(text,verified):
    if type(verified) is not bool:raise ValueError('DeepSeek host verification policy must be boolean')
    manifest=dict(version='dllm-deepseek-host-native-v1',renderer='deepseek-v4-60d8d70770c6776ff598c94bb586a859a38244f1',
        native_tools=True,tokenizer_sha256=hashlib.sha256(text.tokenizer.to_str().encode()).hexdigest(),
        vocab_size=text.vocab_size,eos=sorted(text.eos),verified=verified)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).digest()


def native_session_identity(text,*,verified=False):return _policy(_snapshot(text),verified)


class HostDeepSeekGenerator(NativeHostGenerator):
    model_type='deepseek_v4'
    quantization='fp4_fp8_reference'
    policy=staticmethod(_policy)
    _snapshot_text=staticmethod(_snapshot)
    _validate_text=staticmethod(validate_text_config)

    def _stream(self,codec):
        thinking,allow_tools,_=codec
        return TokenStream(self.text,enable_thinking=thinking,allow_tool_calls=allow_tools)

    def _validate_piece(self,piece,codec):
        if piece.get('tool_calls'):validate_tool_arguments(piece['tool_calls'],codec[2])

    def _prepare(self,conversations,maximum,tools,ragged,options):
        if not isinstance(conversations,list) or not 1<=len(conversations)<=self.max_batch_size:
            raise ValueError('DeepSeek host batch exceeds negotiated slots')
        # Detach declarations once, so prompt and final validation use one contract.
        contract={}
        if tools is not None:
            prepare_tools(tools)
            tools=json.loads(json.dumps(tools,ensure_ascii=False,allow_nan=False))
            contract=prepare_tools(tools)[1]
        prompts,limits=prepare_batch(self.text,conversations,maximum,self.max_context,ragged=ragged,tools=tools,**options)
        if any(set(prompt)&set(self.worker.excluded_ids) for prompt in prompts):
            raise ValueError('DeepSeek host prompt contains excluded IDs')
        return prompts,limits,(options['enable_thinking'],tools is not None,contract)

    def generate(self,messages,*,max_new_tokens=128,enable_thinking=False,drop_thinking=True,tools=None):
        if type(max_new_tokens) is not int:raise ValueError('single DeepSeek host output limit must be an integer')
        prepared=self._prepare([messages],max_new_tokens,tools,False,dict(enable_thinking=enable_thinking,drop_thinking=drop_thinking))
        with closing(self._generate(*prepared)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_batch(self,conversations,*,max_new_tokens=128,enable_thinking=False,drop_thinking=True,tools=None,ragged=False):
        yield from self._generate(*self._prepare(conversations,max_new_tokens,tools,ragged,
            dict(enable_thinking=enable_thinking,drop_thinking=drop_thinking)))

    def generate_tools(self,messages,tools,**options):
        if tools is None:raise ValueError('DeepSeek tools require declarations')
        yield from self.generate(messages,tools=tools,**options)

    def generate_batch_tools(self,conversations,tools,**options):
        if tools is None:raise ValueError('DeepSeek tools require declarations')
        yield from self.generate_batch(conversations,tools=tools,**options)
