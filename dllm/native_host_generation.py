"""Shared native host lifecycle with explicit tokenizer, parser and policy hooks."""
from contextlib import closing
import threading
from tokenizers import Tokenizer,decoders
from .glm_text import CONTROLS,EOS,USER,OBSERVATION
from .glm_generation import TokenStream,validate_text_config
from .glm_flash_tools import FlashTools,_json,_dump
from .host_completion import HostCompletion
from .host_pipeline_worker import HostPipelineCoordinator
from .reference_batch import prepare_batch
from .reference_cancellation import check_cancelled


def snapshot_text(text,text_type):
    if (text_type is None or type(text) is not text_type or type(text.vocab_size) is not int
            or not 1 <= text.vocab_size <= 2**21 or type(text.tokenizer_size) is not int):
        raise ValueError('native host policy requires an explicit native tool/text tokenizer')
    encoded=text.tokenizer.to_str()
    if len(encoded.encode()) > 64*1024**2: raise ValueError('native host tokenizer exceeds metadata bounds')
    tokenizer=Tokenizer.from_str(encoded);tokenizer.no_padding();tokenizer.no_truncation()
    vocab=tokenizer.get_vocab();count=len(vocab)
    if (not isinstance(tokenizer.decoder,decoders.ByteLevel) or not 1<=count<=text.vocab_size
            or set(vocab.values())!=set(range(count)) or count!=text.tokenizer_size
            or any(tokenizer.token_to_id(marker) is None or
                   tokenizer.encode(marker,add_special_tokens=False).ids!=[tokenizer.token_to_id(marker)] for marker in CONTROLS)):
        raise ValueError('invalid native host tokenizer/control mapping')
    eos=frozenset(tokenizer.token_to_id(marker) for marker in (EOS,USER,OBSERVATION))
    if text.eos!=eos: raise ValueError('native host stop policy disagrees with control markers')
    result=text_type.__new__(text_type)
    result.tokenizer,result.vocab_size,result.tokenizer_size,result.eos=tokenizer,text.vocab_size,count,eos
    return result


class NativeHostGenerator:
    """Single owner, up to the negotiated slot count; no HTTP or command admission.

    Healthy completions release slots for reuse. Abandonment, parsing errors and
    cancellation after dispatch discard the host epoch. Keep this owner and call
    cleanup/close again when transport or parser cleanup cannot finish.
    """
    text_type=stream_type=policy=model_type=None
    quantization='fp8_reference'
    input_kind='tokens'

    def _snapshot_text(self,text):return snapshot_text(text,self.text_type)

    def _validate_text(self,config,text,context):validate_text_config(config,text,context)

    def _stream(self,codec):return TokenStream(self.text) if codec is None else self.stream_type(self.text,codec)

    def _validate_piece(self,piece,codec):pass

    def _excluded_matches(self,worker,text):
        return not any(token<text.tokenizer_size for token in worker.excluded_ids)

    def __init__(self,worker,text,*,chunk_size=128):
        text=self._snapshot_text(text)
        verified=getattr(worker,'model_identity_verified',False)
        if (not isinstance(worker,HostPipelineCoordinator) or worker.topology.config.get('model_type')!=self.model_type
                or worker.topology.input_kind!=self.input_kind
                or worker.topology.quantization!=self.quantization
                or worker.topology.session_identity!=self.policy(text,verified)
                or worker.mapped_vocab_size!=text.tokenizer_size or worker.topology.text_config['vocab_size']!=text.vocab_size
                or not self._excluded_matches(worker,text)
                or type(chunk_size) is not int or not 1<=chunk_size<=worker.topology.chunk_size):
            raise ValueError('native host generation requires matching native policy and bounded ready topology')
        self._validate_text(worker.topology.text_config,text,worker.stage.context)
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
        if not self._active.acquire(blocking=False):raise RuntimeError('native host generation is active')
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
            raise ValueError('native host batch exceeds negotiated slots')
        codec=None;tokenize=None
        if tools is not None:
            declarations=_json(_dump(tools));codec=FlashTools(declarations)
            tokenize=lambda messages,**kw:self.text.tool_chat_tokens(messages,declarations,**kw)
        prompts,limits=prepare_batch(self.text,conversations,maximum,self.max_context,
                                    tokenize=tokenize,ragged=ragged,**options)
        if any(set(prompt)&set(self.worker.excluded_ids) for prompt in prompts):
            raise ValueError('native host prompt contains excluded media/control IDs')
        return prompts,limits,codec

    def _generate(self,prompts,limits,codec,*,prepare_embeddings=None):
        if not self._active.acquire(blocking=False):raise RuntimeError('native host generation is active')
        streams=[];finals=[];primary=None;dispatched=False
        try:
            self._retry_cleanup();self.completion._check_cancelled(None);check_cancelled()
            for _ in prompts:streams.append(self._stream(codec))
            calls=[False]*len(prompts)
            def cancelled():check_cancelled();return False
            prepared=prepare_embeddings() if prepare_embeddings is not None else None
            self.completion._check_cancelled(None);check_cancelled()
            complete=(self.completion.generate_batch if prepared is None else self.completion.generate_embedding_batch)
            dispatched=True
            with closing(complete(prompts if prepared is None else prepared,max_new_tokens=limits,
                    chunk_size=self.chunk_size,cancelled=cancelled)) as events:
                for event in events:
                    check_cancelled();row=event['row']
                    pieces=(streams[row].feed(event['token_id']) if 'token_id' in event
                            else streams[row].finish(truncated=event['finish_reason']=='length'))
                    for piece in pieces:
                        self.completion._check_cancelled(None);check_cancelled()
                        self._validate_piece(piece,codec)
                        calls[row]|=bool(piece.get('tool_calls'))
                        yield dict(piece,row=row)
                    if 'finish_reason' in event:
                        finals.append(dict(row=row,finish_reason='tool_calls' if calls[row] else event['finish_reason'],
                            usage={key:event['usage'][key] for key in ('prompt_tokens','completion_tokens')}))
            self.completion._check_cancelled(None)
            if len(finals)!=len(prompts):raise RuntimeError('native host generation did not finish every row')
        except BaseException as error:
            primary=error
            if dispatched:
                try:self.completion.close()
                except BaseException:error.add_note('native host epoch cleanup failed; retain owner and retry close')
            raise
        finally:
            try:
                try:self._release_streams(streams)
                except BaseException as error:
                    if primary is not None:primary.add_note('native host parser cleanup failed; retained for retry')
                    else:
                        try:self.completion.close()
                        except BaseException:error.add_note('native host epoch cleanup also failed')
                        raise
            finally:self._active.release()
        check_cancelled()
        for event in sorted(finals,key=lambda event:event['row']):
            self.completion._check_cancelled(None)
            check_cancelled()
            yield event

    def generate(self,messages,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int:raise ValueError('single native output limit must be an integer')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prepared=self._prepare([messages],max_new_tokens,None,False,options)
        with closing(self._generate(*prepared)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_batch(self,conversations,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False,ragged=False):
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        yield from self._generate(*self._prepare(conversations,max_new_tokens,None,ragged,options))

    def generate_tools(self,messages,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if tools is None:raise ValueError('native tools require function declarations')
        if type(max_new_tokens) is not int:raise ValueError('single native output limit must be an integer')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prepared=self._prepare([messages],max_new_tokens,tools,False,options)
        with closing(self._generate(*prepared)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_batch_tools(self,conversations,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False,ragged=False):
        if tools is None:raise ValueError('native tools require function declarations')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        yield from self._generate(*self._prepare(conversations,max_new_tokens,tools,ragged,options))
