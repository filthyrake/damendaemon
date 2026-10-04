"""Token-framed text Harmony output; channels never become answer text.

This parser handles assistant analysis/final/commentary messages. Declared function handoffs are buffered until complete JSON is available.
"""
import json
import uuid
from .stopping import text_delta
from .tool_calls import _json_object,validate_tools


_MARKERS={
    'start':(200006,('<|start|>',)),
    'message':(200008,('<|message|>',)),
    'end':(200007,('<|end|>',)),
    'return':(200002,('<|return|>','<|fim_suffix|>')),
    'call':(200012,('<|call|>','<|ghissue|>')),
    'channel':(200005,('<|channel|>','<|meta_sep|>')),
    'format':(200003,('<|constrain|>','<|meta_start|>')),
    'eot':(199999,('<|endoftext|>',)),
}


class TextStream:
    def __init__(self,tokenizer,stops=(),tools=None):
        for kind,(expected,names) in _MARKERS.items():
            if not any(tokenizer.token_to_id(name)==expected for name in names):
                raise ValueError(f'unsupported Harmony tokenizer marker: {kind}')
        self.tokenizer=tokenizer;self.stops=stops
        self.markers={value[0]:key for key,value in _MARKERS.items()}
        self.phase='header';self.prefix='assistant';self.header={'role':[]}
        self.field='role';self.header_count=0;self.channel=None;self.body=[]
        self.completed={'content':'','analysis':''};self.emitted={'content':'','analysis':''}
        self.closed=False
        self.names={t["function"]["name"] for t in validate_tools(tools) or []}
        self.recipient=None;self.calls=[]

    def _decode(self,tokens):
        return self.tokenizer.decode(tokens,skip_special_tokens=False)

    def _begin_body(self):
        role=(self.prefix+self._decode(self.header['role'])).split()
        channel=self._decode(self.header.get('channel',[])).split()
        formatting=self._decode(self.header.get('format',[])).strip()
        fields=role[1:]+channel[1:]
        recipients=[x[3:] for x in fields if x.startswith('to=')]
        if (not role or role[0]!='assistant' or len(recipients)>1
                or any(x!='json' and not x.startswith('to=') for x in fields)):
            raise ValueError('invalid Harmony assistant header')
        recipient=recipients[0] if recipients else 'all'
        self.recipient=None
        if recipient!='all':
            if not recipient.startswith('functions.') or recipient[10:] not in self.names:
                raise ValueError('Harmony function recipient is not declared')
            self.recipient=recipient[10:]
        self.channel=channel[0] if channel else 'final'
        if self.channel not in ('analysis','final','commentary') or formatting not in ('','json'):
            raise ValueError('unsupported Harmony output channel or format')
        if self.recipient and self.channel!='commentary':
            raise ValueError('Harmony function calls require the commentary channel')
        self.phase='body';self.body=[]

    def _commit_body(self):
        if self.recipient:raise ValueError('Harmony function call requires a handoff marker')
        key='analysis' if self.channel=='analysis' else 'content'
        self.completed[key]+=self._decode(self.body)
        self.body=[]

    def feed(self,token,final=False):
        if self.closed:raise ValueError('Harmony output continued after its terminal marker')
        marker=self.markers.get(token);terminal=False
        if marker=='start':
            if self.phase!='after':raise ValueError('unexpected Harmony message start')
            self.phase='header';self.prefix='';self.header={'role':[]};self.field='role';self.header_count=0
        elif marker in ('channel','format'):
            if self.phase!='header' or marker in self.header:raise ValueError('unexpected Harmony header metadata')
            self.header[marker]=[];self.field=marker
        elif marker=='message':
            if self.phase!='header':raise ValueError('unexpected Harmony body separator')
            self._begin_body()
        elif marker=='end':
            if self.phase!='body':raise ValueError('unexpected Harmony message end')
            self._commit_body();self.phase='after'
        elif marker in ('return','eot'):
            if self.phase=='body':self._commit_body()
            elif marker=='return':raise ValueError('Harmony return without a message body')
            self.phase='after';terminal=True
        elif marker=='call':
            if self.phase!='body' or not self.recipient:
                raise ValueError('Harmony handoff requires a declared function body')
            arguments=_json_object(self._decode(self.body))
            self.calls.append({'id':'call_'+uuid.uuid4().hex,'type':'function','function':{
                'name':self.recipient,'arguments':json.dumps(arguments,ensure_ascii=False,allow_nan=False)}})
            self.body=[];self.phase='after';terminal=True
        elif token>=199998:
            raise ValueError('unsupported non-text Harmony control token')
        elif self.phase=='header':
            self.header_count+=1
            if self.header_count>256:raise ValueError('Harmony header exceeds token limit')
            self.header[self.field].append(token)
        elif self.phase=='body':
            self.body.append(token)
        else:raise ValueError('Harmony content outside a message')
        decoded=dict(self.completed)
        if self.phase=='body' and not self.recipient:
            key='analysis' if self.channel=='analysis' else 'content'
            decoded[key]+=self._decode(self.body)
        content,self.emitted['content'],stopped=text_delta(decoded['content'],self.emitted['content'],
            self.stops,final or terminal)
        reasoning,self.emitted['analysis'],_=text_delta(decoded['analysis'],self.emitted['analysis'],(),final or terminal)
        self.closed=final or terminal or stopped
        return content,reasoning,terminal or stopped


def output_stream(model,raw_prompt,stops=(),schema=None,logprobs=None,ignore_eos=False,tools=None):
    if raw_prompt is not None or getattr(model,'config',{}).get('model_type')!='gpt_oss':return None
    if schema is not None or logprobs is not None or ignore_eos:
        raise ValueError('Harmony text chat does not yet support structured output, logprobs or ignoring EOS')
    return TextStream(model.tokenizer,stops,tools)
