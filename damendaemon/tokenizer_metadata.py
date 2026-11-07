"""Bounded checkpoint metadata; vocabulary tokenizer.json has a separate contract.

asm-rewrite branch: all logic lives in native/tokenizer_metadata_asm.s
(bridge: tokenizer_metadata_asm.py). The constants are real module
globals — the assembly resolves them via getattr at call time so the
test suite can patch them here.
"""
from pathlib import Path  # noqa: F401  (import compatibility)

MAX_TOKENIZER_CONFIG_BYTES = 16 * 1024 * 1024
MAX_GENERATION_CONFIG_BYTES = 1024 * 1024
MAX_CHECKPOINT_TEMPLATE_BYTES = MAX_TOKENIZER_CONFIG_BYTES

from .safetensor_header import parse_metadata_json  # noqa: F401,E402
from .tokenizer_metadata_asm import (  # noqa: E402
    _read_object,
    checkpoint_template,
    read_generation_config,
    read_tokenizer_config,
    special_token_text,
)
