"""Explicit pinned GLM native policy over shared host generation ownership."""
import hashlib
import json
from .glm_tool_text import GlmToolText
from .glm_tool_output import GlmToolTokenStream
from .native_host_generation import NativeHostGenerator,snapshot_text


def _policy(text,verified):
    if type(verified) is not bool: raise ValueError('GLM host verification policy must be boolean')
    manifest=dict(version='dllm-glm-host-native-v1',renderer='glm53-aca966e4e02791568aa6a4ced368624b3d897f42',
        native_tools=True,tokenizer_sha256=hashlib.sha256(text.tokenizer.to_str().encode()).hexdigest(),
        vocab_size=text.vocab_size,mapped_vocab_size=text.tokenizer_size,eos=sorted(text.eos),verified=verified)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).digest()


def native_session_identity(text,*,verified=False):
    """Bind pinned native format, canonical tokenizer/stops and verification mode."""
    return _policy(snapshot_text(text,GlmToolText),verified)


class HostGlmGenerator(NativeHostGenerator):
    text_type=GlmToolText
    stream_type=GlmToolTokenStream
    model_type='glm_moe_dsa'
    policy=staticmethod(_policy)
