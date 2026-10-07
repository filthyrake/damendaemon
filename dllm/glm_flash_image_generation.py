"""Owned CPU Flash still-image responses; explicit reference API, not serving."""
from .glm_flash_generation import FlashReferenceGenerator
from .glm_flash_image_text import FlashImageText,MEDIA,_render
from .glm_flash_images import prepare_images
from .glm_flash_vision import FlashVision
from .glm_flash_image_embeddings import image_embeddings


class FlashImageReferenceGenerator(FlashReferenceGenerator):
    text_type=FlashImageText

    def __init__(self,model,text,vision,*,max_context=4096,max_cache_bytes=256*1024**2,
                 image_workspace_bytes=64*1024**2,image_output_bytes=64*1024**2):
        super().__init__(model,text,max_context=max_context,max_cache_bytes=max_cache_bytes)
        if (not isinstance(vision,FlashVision) or vision.c['out_hidden_size']!=model.embedding.shape[1]
                or vision.c['in_channels']!=3 or vision.c['patch_size']!=14
                or vision.c['temporal_patch_size']!=2 or vision.c['spatial_merge_size']!=2
                or any(type(v) is not int or not 1<=v<=2**31 for v in (image_workspace_bytes,image_output_bytes))):
            raise ValueError('Flash image generation requires matching prepared still-image vision and language geometry')
        config=getattr(model,'config',None);source=getattr(vision,'source_config',None)
        if config is not None and source is not None and config!=source:raise ValueError('Flash vision and language checkpoint configurations disagree')
        if config is not None and config.get('image_token_id',text.image_token_id)!=text.image_token_id:
            raise ValueError('Flash image token disagrees with model configuration')
        self.vision=vision;self.image_workspace_bytes=image_workspace_bytes;self.image_output_bytes=image_output_bytes
        self.blocked_media=tuple(sorted({token for marker in MEDIA if (token:=text.tokenizer.token_to_id(marker)) is not None}))

    def generate_images(self,messages,images,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:raise ValueError('GLM output length must be 1..4096 tokens')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        _,count=_render(messages,**options)
        if not isinstance(images,(list,tuple)) or not 1<=len(images)==count:
            raise ValueError('Flash image response requires one prepared RGB tensor per image marker')
        patches,grids=prepare_images(images,max_patches=self.vision.max_tokens,max_workspace_bytes=self.image_workspace_bytes,
                                    max_output_bytes=self.image_output_bytes)
        tokens=self.text.image_chat_tokens(messages,grids,**options)
        if not tokens or len(tokens)+max_new_tokens>self.max_context:raise ValueError('GLM prompt and requested output exceed context')
        self.model._input_budget((1,len(tokens)),False)
        def prefill(ids,**limits):
            nonlocal patches
            features=self.vision.forward(patches,grids)
            patches=None  # Release processor output before language cache allocation.
            values=image_embeddings(self.model,ids,[features],image_token_id=self.text.image_token_id)
            return self.model.prefill_embeddings(values,**limits)
        yield from self._generate_tokens(tokens,max_new_tokens,prefill=prefill,blocked_token_ids=self.blocked_media)
