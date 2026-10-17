"""Explicit image-capable model wrapper and expanded prompt accounting."""
import copy

from .vision_inputs import prepare_images, plan_images
from .vision_messages import decode_image_messages
from .vision_positions import image_positions


class ImagePrompt(list):
    def __init__(self,tokens,pixels,grids,identities):
        super().__init__(tokens)
        self.pixels,self.grids,self.identities=pixels,grids,identities


class VisionModelWeights:
    def __init__(self,text,vision):
        self.text=text
        self.vision_weights=vision
        from .vision_policy import validate_model
        validate_model(text.config, text.max_context)
        self.bytes=text.bytes+vision.bytes

    def __getattr__(self,name):
        return getattr(self.text,name)

    def chat_tokens(self,messages,**options):
        prepared=decode_image_messages(messages)
        if not prepared.images:return self.text.chat_tokens(prepared.messages,**options)
        if options.get('tools') or options.get('tool_json'):
            raise ValueError('image requests with tool calls are not yet qualified')
        # An isolated render view permits image markers without weakening the
        # text model's raw-prompt or plain-text rejection policy.
        view=copy.copy(self.text)
        view.config={**self.text.config,'multimodal_token_ids':[]}
        tokens=view.chat_tokens(prepared.messages,**options)
        count=len(prepared.images)
        if (any(tokens.count(marker)!=count for marker in (248053,248056,248054))
                or 248057 in tokens):
            raise ValueError('image markers do not match supplied images')
        shapes = plan_images(prepared.images)
        grids = [[1, height//16, width//16] for height, width in shapes]
        expanded_count = len(tokens) - count + sum(height*width//1024 for height, width in shapes)
        if expanded_count > self.max_context:
            raise ValueError('expanded image prompt exceeds configured context')
        expanded=[]
        image=0
        for token in tokens:
            if token==248056:
                _,height,width=grids[image]
                expanded.extend([token]*(height*width//4))
                image+=1
            else:expanded.append(token)
        image_positions(expanded,grids)
        pixels,grids=prepare_images(prepared.images)
        return ImagePrompt(expanded,pixels,grids,prepared.identities)
