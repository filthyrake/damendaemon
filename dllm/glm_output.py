"""Bounded incremental ordinary GLM reasoning/content output; no tool execution."""
from .glm_text import CONTROLS,EOS,USER,OBSERVATION,END_THINK,MAX_TEXT_BYTES

MARKERS=CONTROLS+('<tool_call>','</tool_call>','<tool_response>','</tool_response>',
                  '<arg_key>','</arg_key>','<arg_value>','</arg_value>',
                  '<|begin_of_image|>','<|image|>','<|video|>','<|begin_of_audio|>')
STOPS=(EOS,USER,OBSERVATION)


class GlmOutput:
    def __init__(self):
        self.channel='reasoning_content';self.pending='';self.bytes=0
        self.ended=self.closed=self.failed=False

    def _usable(self):
        if self.closed or self.failed:raise RuntimeError('GLM output parser is closed or failed')

    def feed(self,text):
        self._usable()
        try:
            if not isinstance(text,str):raise ValueError('GLM completion chunks must be text')
            self.bytes+=len(text.encode('utf-8'))
            if self.bytes>MAX_TEXT_BYTES:raise ValueError('GLM completion exceeds byte limit')
            if self.ended and text:raise ValueError('text after GLM stop marker')
            self.pending+=text;events=[]
            while self.pending:
                matches=[(self.pending.find(marker),marker) for marker in MARKERS if marker in self.pending]
                if not matches:
                    keep=0
                    for size in range(1,min(len(self.pending),max(map(len,MARKERS))-1)+1):
                        if any(marker.startswith(self.pending[-size:]) for marker in MARKERS):keep=size
                    boundary=len(self.pending)-keep
                    if boundary:events.append({self.channel:self.pending[:boundary]})
                    self.pending=self.pending[boundary:];break
                position,marker=min(matches,key=lambda pair:pair[0])
                if position:events.append({self.channel:self.pending[:position]})
                self.pending=self.pending[position+len(marker):]
                if marker==END_THINK and self.channel=='reasoning_content':self.channel='content'
                elif marker in STOPS and self.channel=='content':
                    self.ended=True
                    if self.pending:raise ValueError('text after GLM stop marker')
                else:raise ValueError('unsupported or malformed GLM completion control marker')
            return events
        except BaseException:
            self.failed=True;self.pending='';raise

    def finish(self,*,truncated=False):
        self._usable()
        if type(truncated) is not bool:raise ValueError('GLM truncation flag must be boolean')
        if not self.ended and not truncated:
            self.failed=True;self.pending=''
            raise ValueError('incomplete GLM completion')
        events=[{self.channel:self.pending}] if self.pending else []
        self.pending='';self.closed=True
        return events
