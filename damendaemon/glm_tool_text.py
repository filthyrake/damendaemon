"""Explicit GLM-5.3 text/tool adapter over verified shared bounded formatting.

Six tool-history prompt/token cases independently match the GLM-5.3 template at
aca966e4e02791568aa6a4ced368624b3d897f42 and the Flash subset. Reuse excludes
media and deferred-tool branches. See docs/licenses/glm53-LICENSE.txt and
docs/licenses/glm53-flash-MIT.txt; tests/fixtures/glm53_tool_prompt_contract.json
records the separately pinned GLM template digest and official token IDs.

Delegation shim — decision logic lives in native/glm_tool_text_asm.s.
"""

from .glm_tool_text_asm import (GlmToolText, GlmText, MAX_TOKENS, render_tool_chat)
