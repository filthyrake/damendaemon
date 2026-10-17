"""Bounded native request preflight for the explicit GLM/Flash host commands."""
from dataclasses import dataclass
from damendaemon.cpu_reference_text import read_json
from damendaemon.reference_batch import prepare_batch


@dataclass(frozen=True)
class NativeRequest:
    messages: list
    tools: object
    batch: bool
    ragged: bool
    options: dict
    maximum: int
    prompts: list

    def generate(self,runner):
        options=dict(self.options,max_new_tokens=self.maximum)
        if self.batch:options['ragged']=self.ragged
        if self.tools is not None:
            generate=runner.generate_batch_tools if self.batch else runner.generate_tools
            return generate(self.messages,self.tools,**options)
        generate=runner.generate_batch if self.batch else runner.generate
        return generate(self.messages,**options)


def validate_mode(args):
    native=getattr(args,'native_chat',False)
    fields=('messages','batch_messages','tools','reasoning_effort')
    request_options=any(getattr(args,name,None) is not None for name in fields)
    request_options|=any(getattr(args,name,False) for name in ('ragged_batch','clear_history_reasoning','thinking','keep_history_reasoning'))
    if not native and request_options:raise ValueError('native host options require --native-chat')
    if native and (args.tokens is not None or args.batch_inputs is not None):
        raise ValueError('native host chat cannot accept raw token inputs')
    if args.role=='worker' and request_options:raise ValueError('reference worker cannot accept native request options')
    if args.role=='serve':
        if not native or request_options:raise ValueError('reference serve requires native chat and no request options')
        if (not 0<=args.queue_size<=64 or (args.queue_size==0 and (args.queue_timeout!=120 or args.stop_after_drain))
                or args.client_decode_turns or args.client_prefill_turns or args.interleaved_prefill or args.idle_timeout!=0):
            raise ValueError('reference host HTTP needs an explicit queue for drain; decode/prefill fairness and idle-stop are unsupported')
    return native


def prepare_request(args,text,*,family='glm'):
    if args.role=='worker':return None
    if sum(value is not None for value in (args.prompt,args.messages,args.batch_messages))!=1:
        raise ValueError('native host completion requires one prompt or messages source')
    if args.ragged_batch and args.batch_messages is None:raise ValueError('ragged native requests require batch messages')
    if not 1<=args.prefill_chunk<=4096:raise ValueError('native host prefill chunk must be 1..4096')
    messages=([dict(role='user',content=args.prompt)] if args.prompt is not None
              else read_json(args.batch_messages or args.messages,1024**2))
    tools=read_json(args.tools,1024**2) if args.tools is not None else None
    if args.tools is not None and tools is None:raise ValueError('native tools require function declarations')
    batch=args.batch_messages is not None;rows=messages if batch else [messages]
    if not isinstance(rows,list) or not 1<=len(rows)<=args.slots:raise ValueError('native host batch exceeds host slots')
    if family=='deepseek':
        options=dict(enable_thinking=args.thinking,drop_thinking=not args.keep_history_reasoning,tools=tools)
        tokenize=None
    else:
        options=dict(reasoning_effort=args.reasoning_effort,clear_thinking=args.clear_history_reasoning)
        tokenize=(lambda row,**kw:text.tool_chat_tokens(row,tools,**kw)) if tools is not None else None
    prompts,_=prepare_batch(text,rows,args.max_new_tokens,args.context,tokenize=tokenize,ragged=args.ragged_batch,**options)
    if family=='deepseek':options.pop('tools')
    return NativeRequest(messages,tools,batch,args.ragged_batch,options,args.max_new_tokens,prompts)
