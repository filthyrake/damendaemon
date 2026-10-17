"""Incremental Flash ordinary channels plus atomic schema-bound tool calls."""
from io import StringIO

from .glm_output import GlmOutput, STOPS
from .glm_text import MAX_TEXT_BYTES, CONTROLS
from .glm_generation import TokenStream
from .bytelevel_stream import ByteLevelStream
from .glm_flash_tools import FlashTools, OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE, _text
from .glm_flash_tool_text import FlashToolText


class FlashToolOutput:
    def __init__(self,codec):
        if not isinstance(codec,FlashTools):raise ValueError('Flash tool output requires prepared declarations')
        self.codec=codec;self.ordinary=GlmOutput();self.pending='';self.buffer=None
        self.bytes=0;self.closed=False;self.failed=False

    def _usable(self):
        if self.closed or self.failed:raise RuntimeError('Flash tool output is closed or failed')

    def close(self):
        self.closed=True;self.pending=''
        if self.buffer is not None:self.buffer.close()

    def feed(self,text):
        self._usable()
        try:
            _text(text);self.bytes+=len(text.encode('utf-8'))
            if self.bytes>MAX_TEXT_BYTES:raise ValueError('Flash tool completion exceeds byte limit')
            if self.buffer is not None:
                self.buffer.write(text);return []
            self.pending+=text
            boundary=self.pending.find(OPEN)
            if boundary>=0:
                events=self.ordinary.feed(self.pending[:boundary])
                if self.ordinary.channel!='content' or self.ordinary.ended:
                    raise ValueError('Flash tool calls require a completed reasoning channel')
                self.buffer=StringIO();self.buffer.write(self.pending[boundary:]);self.pending=''
                return events
            keep=0
            for length in range(1,min(len(self.pending),len(OPEN)-1)+1):
                if OPEN.startswith(self.pending[-length:]):keep=length
            boundary=len(self.pending)-keep
            events=self.ordinary.feed(self.pending[:boundary]) if boundary else []
            self.pending=self.pending[boundary:]
            return events
        except BaseException:
            self.failed=True;self.close();raise

    def finish(self,*,truncated=False):
        self._usable()
        try:
            if type(truncated) is not bool:raise ValueError('Flash truncation flag must be boolean')
            if self.buffer is not None:
                value=self.buffer.getvalue()
                stop=next((marker for marker in STOPS if value.endswith(marker)),None)
                if stop is None:raise ValueError('incomplete Flash tool completion')
                return [dict(tool_calls=self.codec.parse(value[:-len(stop)]))]
            if self.pending:raise ValueError('incomplete Flash tool opener')
            return self.ordinary.finish(truncated=truncated)
        except BaseException:
            self.failed=True;raise
        finally:self.close()


class FlashToolTokenStream(TokenStream):
    def __init__(self,text,codec):
        if not isinstance(text,FlashToolText):raise ValueError('Flash tool stream requires explicit tool tokenizer')
        super().__init__(text)
        self.decoder=ByteLevelStream(text.tokenizer,CONTROLS+(OPEN,CLOSE,KEY,END_KEY,VALUE,END_VALUE))
        self.parser=FlashToolOutput(codec)

    def close(self):self.parser.close()
