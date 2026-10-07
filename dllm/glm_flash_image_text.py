"""Explicit pinned Flash still-image chat; no URL/file decoding or video admission."""
from .glm_flash_text import FlashText
from .glm_text import render_chat,MAX_TEXT_BYTES,MAX_TOKENS

BEGIN_IMAGE='<|begin_of_image|>'
IMAGE='<|image|>'
END_IMAGE='<|end_of_image|>'
MEDIA=(BEGIN_IMAGE,IMAGE,END_IMAGE,'<|begin_of_video|>','<|video|>','<|end_of_video|>',
       '<|begin_of_audio|>','<|end_of_audio|>')


def _render(messages,**options):
    if not isinstance(messages,list) or not 1<=len(messages)<=4096:raise ValueError('invalid Flash image conversation')
    converted=[];image_count=0;part_count=0;used_bytes=0
    def text(value):
        nonlocal used_bytes
        if not isinstance(value,str) or any(marker in value for marker in MEDIA):raise ValueError('Flash image chat requires text without literal media markers')
        used_bytes+=len(value.encode('utf-8'))
        if used_bytes>MAX_TEXT_BYTES:raise ValueError('Flash image conversation exceeds text byte limit')
        return value
    for message in messages:
        if not isinstance(message,dict):raise ValueError('invalid Flash image message')
        item=dict(message);content=item.get('content')
        if isinstance(content,list):
            if item.get('role')!='user' or not content:raise ValueError('Flash image parts require nonempty user content')
            parts=[];part_count+=len(content)
            if part_count>4096:raise ValueError('Flash image conversation exceeds part limit')
            for part in content:
                if not isinstance(part,dict):raise ValueError('Flash image parts must be typed mappings')
                if set(part)=={'type','text'} and part.get('type')=='text':parts.append(text(part['text']))
                elif set(part)=={'type'} and part.get('type')=='image':
                    image_count+=1
                    if image_count>256:raise ValueError('Flash image conversation exceeds image limit')
                    parts.append(BEGIN_IMAGE+IMAGE+END_IMAGE)
                else:raise ValueError('unsupported Flash image part; only prepared image markers and text are accepted')
            item['content']=''.join(parts)
        else:item['content']=text(content)
        if 'reasoning_content' in item:text(item['reasoning_content'])
        converted.append(item)
    return render_chat(converted,**options),image_count


def render_image_chat(messages,**options):
    """Render the pinned ordinary reasoning format with explicit still-image markers."""
    return _render(messages,**options)[0]


class FlashImageText(FlashText):
    """Separate image method; inherited ordinary chat still rejects multipart input.

    Image marker order must match the caller's prepared image/feature order.
    The adapter binds counts, not semantic identity or processor provenance.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        ids=[]
        for marker in (BEGIN_IMAGE,IMAGE,END_IMAGE):
            token=self.tokenizer.token_to_id(marker)
            if token is None or self.tokenizer.encode(marker,add_special_tokens=False).ids!=[token]:
                raise ValueError('Flash image markers must be atomic tokenizer tokens')
            ids.append(token)
        if len(set(ids))!=3:raise ValueError('Flash image marker IDs must be distinct')
        self.image_token_id=ids[1]

    def image_chat_tokens(self,messages,image_grids,**options):
        prompt,count=_render(messages,**options)
        if not isinstance(image_grids,(list,tuple)) or len(image_grids)!=count:
            raise ValueError('Flash image marker/grid count mismatch')
        lengths=[]
        for grid in image_grids:
            if (not isinstance(grid,(list,tuple)) or len(grid)!=3 or any(type(n) is not int for n in grid)
                    or grid[0]!=1 or not 2<=grid[1]<=65536 or not 2<=grid[2]<=65536 or grid[1]%2 or grid[2]%2):
                raise ValueError('Flash still-image grids require one frame and even patch dimensions')
            lengths.append(grid[1]*grid[2]//4)
        if sum(lengths)>MAX_TOKENS:raise ValueError('Flash expanded image token budget exceeded')
        ids=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if ids.count(self.image_token_id)!=count:raise ValueError('Flash tokenizer image marker count disagrees')
        if len(ids)-count+sum(lengths)>MAX_TOKENS:raise ValueError('Flash expanded prompt exceeds token limit')
        expanded=[];sizes=iter(lengths)
        for token in ids:
            if token==self.image_token_id:expanded.extend([token]*next(sizes))
            else:expanded.append(token)
        return expanded
