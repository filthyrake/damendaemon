"""Flash inline-image message adapter over the shared bounded PNG/JPEG decoder."""
from .glm_flash_image_text import _render
from .vision_messages import decode_image_messages,ImageMessages


def decode_inline_images(messages,*,max_images=4,max_bytes=8*1024**2,max_pixels=16*1024**2):
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
    _render(prepared)  # Validate Flash structure before decoding any pixels.
    decoded=decode_image_messages(messages,max_images=max_images,max_bytes=max_bytes,max_pixels=max_pixels)
    # The shared decoder's Qwen-rendered message strings are intentionally not
    # reused; Flash retains its own typed markers and pinned template semantics.
    return ImageMessages(prepared,decoded.images,decoded.identities)
