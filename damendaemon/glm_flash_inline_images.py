"""Flash inline-image message adapter over the shared bounded PNG/JPEG decoder."""
from .glm_flash_image_text import _render
import copy
from .vision_messages import decode_image_messages,validate_image_messages,ImageMessages


def _prepare(messages,tools):
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:raise ValueError('invalid Flash inline-image conversation')
    prepared=[]
    for message in messages:
        if not isinstance(message,dict):raise ValueError('invalid Flash inline-image message')
        item=dict(message);content=item.get('content')
        if isinstance(content,list):
            item['content']=[{'type':'image'} if isinstance(part,dict) and set(part)=={'type','image_url'}
                              and part.get('type')=='image_url' else dict(part) if isinstance(part,dict) else part
                              for part in content]
        prepared.append(item)
    if tools is None:
        _render(prepared)
    else:
        from .glm_flash_image_tool_text import render_image_tool_chat
        render_image_tool_chat(prepared,tools)
    # Flash owns history validation. Only image-bearing user content reaches the
    # generic pixel decoder, so its unrelated tool-name conventions do not leak
    # into this explicitly validated Flash history contract.
    carriers=[dict(role='user',content=m['content']) for m in messages if isinstance(m.get('content'),list)]
    return prepared,carriers or [dict(role='user',content='')]


def validate_inline_images(messages,*,tools=None):
    """Validate family history and inline framing without decoding pixels."""
    _,carriers=_prepare(messages,tools)
    validate_image_messages(carriers)
    return copy.deepcopy(messages)


def decode_inline_images(messages,*,tools=None,max_images=4,max_bytes=8*1024**2,max_pixels=16*1024**2):
    prepared,carriers=_prepare(messages,tools)
    decoded=decode_image_messages(carriers,
                                  max_images=max_images,max_bytes=max_bytes,max_pixels=max_pixels)
    # The shared decoder's Qwen-rendered message strings are intentionally not
    # reused; Flash retains its own typed markers and pinned template semantics.
    return ImageMessages(prepared,decoded.images,decoded.identities)
