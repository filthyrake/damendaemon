"""Explicit pinned Flash native text/tool policy for prepared CPU host stages."""
import hashlib
import json
from .glm_flash_tool_text import FlashToolText
from .glm_flash_tool_output import FlashToolTokenStream
from .glm_flash_image_text import MEDIA
from .native_host_generation import NativeHostGenerator,snapshot_text


def native_excluded_ids(text):
    return tuple(sorted({token for marker in MEDIA if (token:=text.tokenizer.token_to_id(marker)) is not None}))


def _policy(text,verified):
    if type(verified) is not bool:raise ValueError('Flash host verification policy must be boolean')
    manifest=dict(version='dllm-flash-host-native-v1',renderer='glm53-flash-eb9eb208eb0d988989d07a6a12d0fdeb5f52574a',
        native_tools=True,tokenizer_sha256=hashlib.sha256(text.tokenizer.to_str().encode()).hexdigest(),
        vocab_size=text.vocab_size,mapped_vocab_size=text.tokenizer_size,eos=sorted(text.eos),
        excluded_ids=native_excluded_ids(text),verified=verified)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).digest()


def native_session_identity(text,*,verified=False):
    return _policy(snapshot_text(text,FlashToolText),verified)


class HostFlashGenerator(NativeHostGenerator):
    text_type=FlashToolText
    stream_type=FlashToolTokenStream
    model_type='glm5_next'
    policy=staticmethod(_policy)

    def _excluded_matches(self,worker,text):
        from .host_topology import HostTopology
        plan=worker.topology
        expected=HostTopology(plan.config,plan.ranges,plan.source_identity,quantization=plan.quantization,
            context=plan.context,context_limit=plan.context_limit,slots=plan.slots,chunk_size=plan.chunk_size,
            frame_bytes=plan.frame_bytes,mapped_vocab_size=text.tokenizer_size,excluded_ids=native_excluded_ids(text),
            session_identity=plan.session_identity)
        return plan.excluded_ids==expected.excluded_ids
