"""Incremental ordinary V4 completion parsing; tool/DSML output fails closed."""
from .deepseek_text import BOS, EOS, USER, ASSISTANT, THINK, END_THINK, MAX_TEXT_BYTES

MARKERS=(BOS,EOS,USER,ASSISTANT,THINK,END_THINK,'<｜DSML｜','｜DSML｜')


class DeepSeekOutput:
    """Emit content/reasoning deltas without retaining the full completion.

    Keep only a possible control-marker suffix between feeds. Already returned
    deltas cannot be retracted if a later chunk is malformed; callers must signal
    the failure rather than treating partial output as a completed answer.
    """
    def __init__(self,*,enable_thinking=False):
        if type(enable_thinking) is not bool:
            raise ValueError('thinking mode must be boolean')
        self.channel='reasoning_content' if enable_thinking else 'content'
        self.pending=''
        self.bytes=0
        self.ended=self.closed=self.failed=False

    def _usable(self):
        if self.closed or self.failed:
            raise RuntimeError('V4 output parser is closed or failed')

    def feed(self,text):
        self._usable()
        try:
            if not isinstance(text,str):raise ValueError('completion chunks must be text')
            self.bytes+=len(text.encode('utf-8'))
            if self.bytes > MAX_TEXT_BYTES:raise ValueError('completion exceeds text byte limit')
            if self.ended and text:raise ValueError('unexpected text after V4 EOS')
            self.pending+=text
            events=[]
            while self.pending:
                matches=[(self.pending.find(marker),marker) for marker in MARKERS if marker in self.pending]
                if not matches:
                    keep=0
                    for size in range(1,min(len(self.pending),max(map(len,MARKERS))-1)+1):
                        if any(marker.startswith(self.pending[-size:]) for marker in MARKERS):keep=size
                    visible=self.pending[:-keep] if keep else self.pending
                    if visible:events.append({self.channel:visible})
                    self.pending=self.pending[-keep:] if keep else ''
                    break
                position,marker=min(matches,key=lambda item:item[0])
                if position:events.append({self.channel:self.pending[:position]})
                self.pending=self.pending[position+len(marker):]
                if marker == END_THINK and self.channel == 'reasoning_content':
                    self.channel='content'
                elif marker == EOS and self.channel == 'content':
                    self.ended=True
                    if self.pending:raise ValueError('unexpected text after V4 EOS')
                else:
                    raise ValueError('unsupported or malformed V4 completion control marker')
            return events
        except BaseException:
            self.failed=True
            self.pending=''
            raise

    def finish(self,*,truncated=False):
        self._usable()
        if type(truncated) is not bool:
            raise ValueError('truncation flag must be boolean')
        if not self.ended and not truncated:
            self.failed=True
            self.pending=''
            raise ValueError('incomplete V4 completion: EOS or thinking terminator missing')
        events=[{self.channel:self.pending}] if self.pending else []
        self.pending=''
        self.closed=True
        return events


def parse_completion(text,*,enable_thinking=False,truncated=False):
    parser=DeepSeekOutput(enable_thinking=enable_thinking)
    result={'content':'','reasoning_content':''}
    for event in parser.feed(text)+parser.finish(truncated=truncated):
        for channel,value in event.items():result[channel]+=value
    return result
