"""Explicit pinned Flash still-image plus tool histories; no raw media controls."""
from .glm_flash_image_text import FlashImageText,BEGIN_IMAGE,IMAGE,END_IMAGE
from .glm_flash_tool_text import FlashToolText,_render_tool_chat
from .glm_flash_tools import _literal
from .glm_text import MAX_TEXT_BYTES


def _render_image_tool_chat(messages,tools,**options):
    count=0;parts_seen=0;used_bytes=0
    def text(value):
        nonlocal used_bytes
        value=_literal(value);used_bytes+=len(value.encode('utf-8'))
        if used_bytes>MAX_TEXT_BYTES:raise ValueError('Flash image/tool text exceeds byte limit')
        return value
    def user_content(content):
        nonlocal count,parts_seen
        if isinstance(content,str):return text(content)
        if not isinstance(content,list) or not content:raise ValueError('Flash image/tool user content needs text or typed parts')
        parts_seen+=len(content)
        if parts_seen>4096:raise ValueError('Flash image/tool part count exceeds limit')
        rendered=[]
        for part in content:
            if not isinstance(part,dict):raise ValueError('invalid Flash image/tool part')
            if set(part)=={'type','text'} and part.get('type')=='text':rendered.append(text(part['text']))
            elif set(part)=={'type'} and part.get('type')=='image':
                count+=1
                if count>256:raise ValueError('Flash image/tool image count exceeds limit')
                rendered.append(BEGIN_IMAGE+IMAGE+END_IMAGE)
            else:raise ValueError('Flash image/tool parts support only prepared still images and text')
        return ''.join(rendered)
    prompt=_render_tool_chat(messages,tools,_user_content=user_content,**options)
    return prompt,count


def render_image_tool_chat(messages,tools,**options):
    return _render_image_tool_chat(messages,tools,**options)[0]


class FlashImageToolText(FlashImageText,FlashToolText):
    def image_tool_chat_tokens(self,messages,image_grids,tools,**options):
        prompt,count=_render_image_tool_chat(messages,tools,**options)
        return self._expand_image_tokens(prompt,count,image_grids)
