"""GLM-5.3 binding for the separately verified schema-bound tool wire format."""
from .glm_generation import TokenStream
from .glm_tool_text import GlmToolText
from .glm_flash_tool_output import FlashToolOutput
from .glm_flash_tools import OPEN,CLOSE,KEY,END_KEY,VALUE,END_VALUE
from .glm_text import CONTROLS
from .bytelevel_stream import ByteLevelStream


class GlmToolTokenStream(TokenStream):
    def __init__(self,text,codec):
        if not isinstance(text,GlmToolText):raise ValueError('GLM tools require explicit tool tokenizer')
        super().__init__(text)
        self.decoder=ByteLevelStream(text.tokenizer,CONTROLS+(OPEN,CLOSE,KEY,END_KEY,VALUE,END_VALUE))
        self.parser=FlashToolOutput(codec)

    def close(self):self.parser.close()
