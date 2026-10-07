"""Explicit native GLM text/tools over a prepared, policy-bound host session."""
from contextlib import closing
import hashlib
import json
import threading
from tokenizers import Tokenizer, decoders
from .glm_text import CONTROLS, EOS, USER, OBSERVATION
from .glm_tool_text import GlmToolText
from .glm_generation import TokenStream, validate_text_config
from .glm_tool_output import GlmToolTokenStream
from .glm_flash_tools import FlashTools, _json, _dump
from .host_completion import HostCompletion
from .host_pipeline_worker import HostPipelineCoordinator
from .reference_batch import prepare_batch
from .reference_cancellation import check_cancelled


def _snapshot(text):
    if (type(text) is not GlmToolText or type(text.vocab_size) is not int
            or not 1 <= text.vocab_size <= 2**21 or type(text.tokenizer_size) is not int):
        raise ValueError('native GLM host policy requires an explicit GLM tool/text tokenizer')
    encoded=text.tokenizer.to_str()
    if len(encoded.encode()) > 64*1024**2: raise ValueError('GLM host tokenizer exceeds metadata bounds')
    tokenizer=Tokenizer.from_str(encoded);tokenizer.no_padding();tokenizer.no_truncation()
    vocab=tokenizer.get_vocab();count=len(vocab)
    if (not isinstance(tokenizer.decoder,decoders.ByteLevel) or not 1<=count<=text.vocab_size
            or set(vocab.values())!=set(range(count)) or count!=text.tokenizer_size
            or any(tokenizer.token_to_id(marker) is None or
                   tokenizer.encode(marker,add_special_tokens=False).ids!=[tokenizer.token_to_id(marker)] for marker in CONTROLS)):
        raise ValueError('invalid native GLM host tokenizer/control mapping')
    eos=frozenset(tokenizer.token_to_id(marker) for marker in (EOS,USER,OBSERVATION))
    if text.eos!=eos: raise ValueError('native GLM host stop policy disagrees with control markers')
    result=GlmToolText.__new__(GlmToolText)
    result.tokenizer,result.vocab_size,result.tokenizer_size,result.eos=tokenizer,text.vocab_size,count,eos
    return result


def _policy(text,verified):
    if type(verified) is not bool: raise ValueError('GLM host verification policy must be boolean')
    manifest=dict(version='dllm-glm-host-native-v1',renderer='glm53-aca966e4e02791568aa6a4ced368624b3d897f42',
        native_tools=True,tokenizer_sha256=hashlib.sha256(text.tokenizer.to_str().encode()).hexdigest(),
        vocab_size=text.vocab_size,mapped_vocab_size=text.tokenizer_size,eos=sorted(text.eos),verified=verified)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).digest()


def native_session_identity(text,*,verified=False):
    """Bind pinned native format, canonical tokenizer/stops and verification mode."""
    return _policy(_snapshot(text),verified)


class HostGlmGenerator:
    """Single owner, up to the negotiated slot count; no HTTP or command admission.

    Healthy completions release slots for reuse. Abandonment, parsing errors and
    cancellation after dispatch discard the host epoch. Keep this owner and call
    cleanup/close again when transport or parser cleanup cannot finish.
    """
    def __init__(self,worker,text,*,chunk_size=128):
        text=_snapshot(text)
        verified=getattr(worker,'model_identity_verified',False)
        if (not isinstance(worker,HostPipelineCoordinator) or worker.topology.config.get('model_type')!='glm_moe_dsa'
                or worker.topology.quantization!='fp8_reference'
                or worker.topology.session_identity!=_policy(text,verified)
                or worker.mapped_vocab_size!=text.tokenizer_size or worker.stage.c['vocab_size']!=text.vocab_size
                or any(token<text.tokenizer_size for token in worker.excluded_ids)
                or type(chunk_size) is not int or not 1<=chunk_size<=worker.topology.chunk_size):
            raise ValueError('GLM host generation requires matching native policy and bounded ready topology')
        validate_text_config(worker.stage.c,text,worker.stage.context)
        self.worker,self.text,self.chunk_size=worker,text,chunk_size
        self.max_context,self.max_batch_size=worker.stage.context,worker.stage.max_batch
        self.failed_cleanups=[];self._active=threading.Lock()
        self.completion=HostCompletion(worker,eos_ids=text.eos)

    def _release_streams(self,streams):
        failures=[]
        for stream in streams:
            close=getattr(stream,'close',None)
            if close is not None:
                try:close()
                except BaseException as error:
                    if all(stream is not pending for pending in self.failed_cleanups):self.failed_cleanups.append(stream)
                    failures.append(error)
        if failures:raise failures[0]

    def _retry_cleanup(self):
        pending=self.failed_cleanups;self.failed_cleanups=[];failure=None
        try:self._release_streams(pending)
        except BaseException as error:failure=error
        if self.completion.stopped.is_set():
            try:self.completion.close()
            except BaseException as error:
                if failure is None:failure=error
        if failure is not None:raise failure

    def cleanup(self):
        if not self._active.acquire(blocking=False):raise RuntimeError('GLM host generation is active')
        try:self._retry_cleanup()
        finally:self._active.release()

    def close(self):
        primary=None
        try:self.completion.close()
        except BaseException as error:primary=error
        if self._active.acquire(blocking=False):
            try:
                try:self._retry_cleanup()
                except BaseException as error:
                    if primary is None:primary=error
            finally:self._active.release()
        if primary is not None:raise primary

    def _prepare(self,conversations,maximum,tools,ragged,options):
        if not isinstance(conversations,list) or not 1<=len(conversations)<=self.max_batch_size:
            raise ValueError('native GLM host batch exceeds negotiated slots')
        codec=None;tokenize=None
        if tools is not None:
            declarations=_json(_dump(tools));codec=FlashTools(declarations)
            tokenize=lambda messages,**kw:self.text.tool_chat_tokens(messages,declarations,**kw)
        prompts,limits=prepare_batch(self.text,conversations,maximum,self.max_context,
                                    tokenize=tokenize,ragged=ragged,**options)
        return prompts,limits,codec

    def _generate(self,prompts,limits,codec):
        if not self._active.acquire(blocking=False):raise RuntimeError('GLM host generation is active')
        streams=[];finals=[];primary=None;dispatched=False
        try:
            self._retry_cleanup();check_cancelled()
            streams=[TokenStream(self.text) if codec is None else GlmToolTokenStream(self.text,codec) for _ in prompts]
            calls=[False]*len(prompts)
            def cancelled():check_cancelled();return False
            dispatched=True
            with closing(self.completion.generate_batch(prompts,max_new_tokens=limits,
                    chunk_size=self.chunk_size,cancelled=cancelled)) as events:
                for event in events:
                    check_cancelled();row=event['row']
                    pieces=(streams[row].feed(event['token_id']) if 'token_id' in event
                            else streams[row].finish(truncated=event['finish_reason']=='length'))
                    for piece in pieces:
                        calls[row]|=bool(piece.get('tool_calls'))
                        yield dict(piece,row=row)
                    if 'finish_reason' in event:
                        finals.append(dict(row=row,finish_reason='tool_calls' if calls[row] else event['finish_reason'],
                            usage={key:event['usage'][key] for key in ('prompt_tokens','completion_tokens')}))
            self.completion._check_cancelled(None)
            if len(finals)!=len(prompts):raise RuntimeError('GLM host generation did not finish every row')
        except BaseException as error:
            primary=error
            if dispatched:
                try:self.completion.close()
                except BaseException:error.add_note('GLM host epoch cleanup failed; retain owner and retry close')
            raise
        finally:
            try:
                try:self._release_streams(streams)
                except BaseException as error:
                    if primary is not None:primary.add_note('GLM host parser cleanup failed; retained for retry')
                    else:
                        try:self.completion.close()
                        except BaseException:error.add_note('GLM host epoch cleanup also failed')
                        raise
            finally:self._active.release()
        check_cancelled()
        for event in sorted(finals,key=lambda event:event['row']):
            self.completion._check_cancelled(None)
            check_cancelled()
            yield event

    def generate(self,messages,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int:raise ValueError('single GLM output limit must be an integer')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prepared=self._prepare([messages],max_new_tokens,None,False,options)
        with closing(self._generate(*prepared)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_batch(self,conversations,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False,ragged=False):
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        yield from self._generate(*self._prepare(conversations,max_new_tokens,None,ragged,options))

    def generate_tools(self,messages,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if tools is None:raise ValueError('native GLM tools require function declarations')
        if type(max_new_tokens) is not int:raise ValueError('single GLM output limit must be an integer')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prepared=self._prepare([messages],max_new_tokens,tools,False,options)
        with closing(self._generate(*prepared)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_batch_tools(self,conversations,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False,ragged=False):
        if tools is None:raise ValueError('native GLM tools require function declarations')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        yield from self._generate(*self._prepare(conversations,max_new_tokens,tools,ragged,options))
