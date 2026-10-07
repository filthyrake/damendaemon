"""Explicit GLM-5.3 text/tool adapter over verified shared bounded formatting.

Six tool-history prompt/token cases independently match the GLM-5.3 template at
aca966e4e02791568aa6a4ced368624b3d897f42 and the Flash subset. Reuse excludes
media and deferred-tool branches. See docs/licenses/glm53-LICENSE.txt and
docs/licenses/glm53-flash-MIT.txt; tests/fixtures/glm53_tool_prompt_contract.json
records the separately pinned GLM template digest and official token IDs.
"""
from .glm_text import GlmText,MAX_TOKENS
from .glm_flash_tool_text import render_tool_chat


class GlmToolText(GlmText):
    def tool_chat_tokens(self,messages,tools,**options):
        prompt=render_tool_chat(messages,tools,**options)
        tokens=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if not tokens or len(tokens)>MAX_TOKENS:raise ValueError('GLM tool prompt token count exceeds bounds')
        return tokens
