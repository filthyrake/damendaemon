"""Incremental V4 text channels with opt-in atomic DSML tool calls."""
from io import StringIO
from .deepseek_tools import START, parse_tool_calls

TOOL_START="\n\n"+START
from .deepseek_text import BOS, EOS, USER, ASSISTANT, THINK, END_THINK, MAX_TEXT_BYTES

MARKERS=(BOS,EOS,USER,ASSISTANT,THINK,END_THINK,'<｜DSML｜','｜DSML｜')


class DeepSeekOutput:
    """Emit content/reasoning deltas without retaining the full completion.

    Keep only a possible control-marker suffix for text; opt-in tool blocks
    are buffered within the completion byte cap and emitted atomically at finish.
    Already returned
    deltas cannot be retracted if a later chunk is malformed; callers must signal
    the failure rather than treating partial output as a completed answer.
    """
    def __init__(self,*,enable_thinking=False,allow_tool_calls=False):
        if type(enable_thinking) is not bool or type(allow_tool_calls) is not bool:
            raise ValueError('thinking mode must be boolean')
        self.allow_tool_calls=allow_tool_calls
        self.markers=MARKERS+(TOOL_START,) if allow_tool_calls else MARKERS
        self.tool_buffer=None
        self.channel='reasoning_content' if enable_thinking else 'content'
        self.pending=''
        self.bytes=0
        self.ended=self.closed=self.failed=False

    def close(self):
        self.closed=True;self.pending=''
        if self.tool_buffer is not None:self.tool_buffer.close()

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
            if self.tool_buffer is not None:
                self.tool_buffer.write(text)
                return []
            self.pending+=text
            events=[]
            while self.pending:
                matches=[(self.pending.find(marker),marker) for marker in self.markers if marker in self.pending]
                keep=0
                for size in range(1,min(len(self.pending),max(map(len,self.markers))-1)+1):
                    if any(marker.startswith(self.pending[-size:]) for marker in self.markers):keep=size
                # A partial tool opener starts before its embedded DSML marker.
                # Keep that prefix instead of prematurely rejecting the marker.
                boundary=len(self.pending)-keep
                if not matches or (keep and boundary < min(position for position,_ in matches)):
                    visible=self.pending[:boundary]
                    if visible:events.append({self.channel:visible})
                    self.pending=self.pending[boundary:]
                    break
                position,marker=min(matches,key=lambda item:item[0])
                if position:events.append({self.channel:self.pending[:position]})
                self.pending=self.pending[position+len(marker):]
                if marker == TOOL_START and self.channel == 'content':
                    self.tool_buffer=StringIO()
                    self.tool_buffer.write(START)
                    self.tool_buffer.write(self.pending)
                    self.pending=''
                    break
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
            if self.tool_buffer is not None:self.tool_buffer.close()
            raise

    def finish(self,*,truncated=False):
        self._usable()
        if type(truncated) is not bool:
            raise ValueError('truncation flag must be boolean')
        if self.tool_buffer is not None:
            try:
                value=self.tool_buffer.getvalue()
                if not value.endswith(EOS):raise ValueError('incomplete V4 tool completion')
                calls=parse_tool_calls(value[:-len(EOS)])
                self.closed=True
                self.ended=True
                return [{'tool_calls':calls}]
            except BaseException:
                self.failed=True
                raise
            finally:
                self.tool_buffer.close()
        if self.allow_tool_calls and self.pending.startswith('\n\n<') and TOOL_START.startswith(self.pending):
            self.failed=True
            self.pending=''
            raise ValueError('incomplete V4 tool opener')
        if not self.ended and not truncated:
            self.failed=True
            self.pending=''
            if self.tool_buffer is not None:self.tool_buffer.close()
            raise ValueError('incomplete V4 completion: EOS or thinking terminator missing')
        events=[{self.channel:self.pending}] if self.pending else []
        self.pending=''
        self.closed=True
        return events


def parse_completion(text,*,enable_thinking=False,truncated=False,allow_tool_calls=False):
    parser=DeepSeekOutput(enable_thinking=enable_thinking,allow_tool_calls=allow_tool_calls)
    result={'content':'','reasoning_content':''}
    if allow_tool_calls:result['tool_calls']=[]
    for event in parser.feed(text)+parser.finish(truncated=truncated):
        for channel,value in event.items():result[channel]+=value
    return result
