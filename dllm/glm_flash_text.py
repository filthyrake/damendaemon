"""Pinned ordinary Flash text adapter; tools and media remain unsupported.

The GLM-5.3-Flash template at eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
has the same ordinary-text branches and tokenizer bytes as the separately pinned
GLM-5.3 contract. Its media branches differ. Reuse only the verified ordinary
format and tokenizer validation, not the media/tool branches. Checkpoint license:
docs/licenses/glm53-flash-MIT.txt. Official prompt/token-ID cases are pinned in
tests/fixtures/glm53_flash_text_contract.json.
"""
from .glm_text import GlmText,render_chat


class FlashText(GlmText):
    """Explicit Flash family binding over the verified shared ordinary format."""
