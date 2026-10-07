"""Explicit single-request CPU reference chat runtime; no accelerator admission."""
from contextlib import closing
import threading
import time
import uuid

from .deepseek_generation import ReferenceGenerator as DeepSeekGenerator
from .deepseek_pipeline_generation import DeepSeekPipelineReferenceGenerator
from .deepseek_tool_prompt import prepare_tools as prepare_deepseek_tools
from .glm_tool_generation import GlmToolReferenceGenerator
from .glm_pipeline_generation import PipelineToolReferenceGenerator
from .glm_flash_pipeline_generation import FlashPipelineToolReferenceGenerator
from .glm_flash_pipeline_image_generation import FlashPipelineImageToolReferenceGenerator
from .glm_flash_tool_generation import FlashToolReferenceGenerator
from .glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from .glm_flash_tools import FlashTools
from .runtime import BusyError
from .reference_cancellation import cancellation_scope,ReferenceCancelled
from .sampling import Sampling
from .tool_calls import validate_tools,validate_messages


class ReferenceRuntime:
    """Own a prepared explicit runner. One request at a time, without a queue.

    On abandonment, close the returned iterator. close() prevents new work and
    asks an active iterator to stop at its next event boundary; it raises BusyError
    while that iterator still owns execution. Retain and retry after it exits.
    Failed cleanup keeps the runner reachable and blocks new prefill.
    """
    native_tool_calls=True

    def __init__(self,runner,alias='model'):
        if type(runner) not in (DeepSeekGenerator,DeepSeekPipelineReferenceGenerator,GlmToolReferenceGenerator,PipelineToolReferenceGenerator,FlashPipelineToolReferenceGenerator,FlashToolReferenceGenerator,FlashImageToolReferenceGenerator,FlashPipelineImageToolReferenceGenerator):
            raise ValueError('CPU HTTP needs an explicit prepared text/tool reference runner')
        if not isinstance(alias,str) or not 1<=len(alias)<=128:raise ValueError('invalid CPU reference alias')
        self.runner=runner;self.alias=alias;self.lock=threading.Lock();self.stopping=threading.Event()
        self.images=type(runner) in (FlashImageToolReferenceGenerator,FlashPipelineImageToolReferenceGenerator)
        self.family='deepseek_v4' if isinstance(runner,DeepSeekGenerator) else 'glm5_next' if self.images or isinstance(runner,FlashToolReferenceGenerator) else 'glm_moe_dsa'
        self.stage_layers=tuple((stage.start,stage.end) for stage in runner.model.stages) if type(runner) in (DeepSeekPipelineReferenceGenerator,PipelineToolReferenceGenerator,FlashPipelineToolReferenceGenerator,FlashPipelineImageToolReferenceGenerator) else ()
        self.context=runner.max_context;self.active=False;self.closed=False
        self.owner=object()
        with runner.lock:
            if getattr(runner,'_http_owner',None) is not None:raise ValueError('CPU reference runner already belongs to HTTP')
            runner._http_owner=self.owner

    def execution_status(self):
        return dict(execution='local_cpu_pipeline',stage_layers=[list(pair) for pair in self.stage_layers]) if self.stage_layers else {}

    def model_ids(self):return [self.alias]
    def model_manifest(self):raise ValueError('CPU reference cannot be exported as a production GPU model manifest')
    def embed(self,*args,**kwargs):raise ValueError('CPU reference embeddings are unsupported')
    def score(self,*args,**kwargs):raise ValueError('CPU reference scoring is unsupported')
    def rerank(self,*args,**kwargs):raise ValueError('CPU reference reranking is unsupported')

    def cleanup_pending(self):
        runner=self.runner
        return bool(runner is not None and (runner.failed_cleanups
            or getattr(runner.model,'failed_cleanups',[]) or getattr(runner.model,'failed_prefills',[])
            or getattr(runner.model,'prefill_cleanup_pending',False)
            or getattr(runner.model,'cleanup_pending',False)))

    def is_healthy(self):
        return not self.stopping.is_set() and not self.cleanup_pending()

    def manage(self,action,**options):
        if action!='status' or options:raise ValueError('CPU reference model management is unsupported')
        return dict(device='cpu',scope='cpu_reference',official_model_qualified=False,model_type=self.family,
            context=self.context,modality='still_image' if self.images else 'text',models=[dict(id=self.alias,residency='closed' if self.closed else 'cpu')],
            active=self.active,closed=self.closed,closing=self.stopping.is_set(),cleanup_pending=self.cleanup_pending(),max_batch_size=1,**self.execution_status())

    def prepare_tools(self,tools,choice=None):
        tools=validate_tools(tools)
        if choice is None:choice='auto' if tools else 'none'
        if choice=='none':return None,None
        if choice!='auto' or not tools:raise ValueError('CPU native tools support only auto or none')
        if self.family=='deepseek_v4':prepare_deepseek_tools(tools)
        else:FlashTools(tools)
        return tools,None

    def prepare_messages(self,messages,*,tools=None):
        if self.images:
            from .glm_flash_inline_images import validate_inline_images
            return validate_inline_images(messages,tools=tools)
        return validate_messages(messages)

    def _cleanup(self):
        if self.runner is not None:self.runner.cleanup()
        if self.stopping.is_set():self.runner=None;self.closed=True

    def close(self):
        self.stopping.set()
        if not self.lock.acquire(blocking=False):raise BusyError('CPU reference shutdown is waiting for its active iterator')
        try:self._cleanup()
        finally:self.lock.release()

    def generate(self,alias,messages,max_tokens=128,*,enable_thinking=None,reasoning_effort=None,**options):
        if alias!=self.alias:raise ValueError('unknown CPU reference model')
        if type(max_tokens) is not int or not 1<=max_tokens<=4096:raise ValueError('CPU reference output limit must be 1..4096')
        tools,_=self.prepare_tools(options.pop('tools',None),options.pop('tool_choice',None))
        response=options.pop('response_format',None);stops=options.pop('stop',None)
        if (response not in (None,{'type':'text'}) or stops
                or set(options)-set(Sampling.__dataclass_fields__) or Sampling(**options)!=Sampling()):
            raise ValueError('CPU reference supports default greedy chat only')
        messages=self.prepare_messages(messages,tools=tools)
        if self.family=='deepseek_v4':
            if reasoning_effort is not None or (enable_thinking is not None and type(enable_thinking) is not bool):
                raise ValueError('unsupported DeepSeek HTTP reasoning options')
            kwargs=dict(enable_thinking=False if enable_thinking is None else enable_thinking,tools=tools)
        else:
            if enable_thinking is not None:raise ValueError('GLM thinking boolean is unsupported; use reasoning_effort')
            if reasoning_effort not in (None,'low','high'):raise ValueError('unsupported GLM HTTP reasoning effort')
            kwargs=dict(reasoning_effort=reasoning_effort)
        if self.stopping.is_set():raise BusyError('CPU reference runtime is closing')
        if not self.lock.acquire(blocking=False):raise BusyError('CPU reference runtime is busy')
        primary=None;started=time.perf_counter();final=None
        try:
            if self.stopping.is_set():raise BusyError('CPU reference runtime is closing')
            self.active=True;runner=self.runner
            if self.images:
                from .glm_flash_inline_images import decode_inline_images
                decoded=decode_inline_images(messages,tools=tools)
                if not decoded.images:raise ValueError('Flash image HTTP requires at least one inline still image')
                if tools:
                    iterator=runner.generate_image_tools(decoded.messages,decoded.images,tools,max_new_tokens=max_tokens,**kwargs)
                else:iterator=runner.generate_images(decoded.messages,decoded.images,max_new_tokens=max_tokens,**kwargs)
            elif self.family!='deepseek_v4' and tools:
                iterator=runner.generate_tools(messages,tools,max_new_tokens=max_tokens,**kwargs)
            else:iterator=runner.generate(messages,max_new_tokens=max_tokens,**kwargs)
            with cancellation_scope(self.stopping.is_set),closing(iterator) as stream:
                for event in stream:
                    if self.stopping.is_set():raise BusyError('CPU reference runtime closed during generation')
                    if 'finish_reason' in event:final=event
                    else:
                        output=dict(text=event.get('content',''))
                        if event.get('reasoning_content'):output['reasoning_content']=event['reasoning_content']
                        if event.get('tool_calls'):
                            output['tool_calls']=[dict(call,id='call_'+uuid.uuid4().hex) for call in event['tool_calls']]
                        yield output
            if final is None:raise RuntimeError('CPU reference ended without a completion result')
        except ReferenceCancelled as error:
            primary=error
            raise BusyError('CPU reference generation cancelled') from error
        except BaseException as error:primary=error;raise
        finally:
            try:
                try:self._cleanup()
                except BaseException:
                    if primary is None:raise
                    primary.add_note('CPU reference HTTP cleanup failed; retain runtime and retry close')
            finally:self.active=False;self.lock.release()
        usage=dict(final['usage']);usage['total_tokens']=usage['prompt_tokens']+usage['completion_tokens']
        yield dict(finish_reason=final['finish_reason'],usage=usage,timing=dict(source_tier='cpu_reference',
            kernel_backend=self.family+'-cpu-reference',official_model_qualified=False,
            total_seconds=time.perf_counter()-started,max_batch_size=1,**self.execution_status()))


def handler_for_reference(runtime,auth=None):
    """Ordinary HTTP policy with explicit reference-runtime health state."""
    if not isinstance(runtime,ReferenceRuntime):raise ValueError('reference handler requires reference runtime')
    from .server import handler_for
    base=handler_for(runtime,auth)
    class Handler(base):
        def do_GET(self):
            if self.path!='/health':return super().do_GET()
            if not self.authorize():return
            healthy=runtime.is_healthy()
            self.send_json(200 if healthy else 503,dict(status='ok' if healthy else 'closed',
                scope='cpu_reference',official_model_qualified=False))
    return Handler
