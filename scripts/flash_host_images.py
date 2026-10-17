"""Explicit coordinator-only Flash inline-image host command admission."""
from dataclasses import dataclass
from damendaemon.cpu_reference_text import read_json
from damendaemon.glm_flash_inline_images import decode_inline_images
from damendaemon.glm_flash_images import plan_images
from damendaemon.glm_flash_tools import FlashTools


_FIELDS=('vision_weight_mib','image_workspace_mib','image_output_mib','max_patches')


def validate_options(args,*,family):
    enabled=getattr(args,'flash_images',False)
    supplied=any(getattr(args,name,None) is not None for name in _FIELDS)
    if not enabled:
        if supplied:raise ValueError('image budgets require --flash-images')
        return False
    if family!='flash' or not args.native_chat or args.role not in ('complete','worker','serve'):
        raise ValueError('Flash images require explicit native Flash complete/worker/serve mode')
    if args.batch_messages is not None or args.ragged_batch or args.prompt is not None:
        raise ValueError('Flash host images require one inline messages request, not prompt/batch input')
    if args.role=='worker' and supplied:raise ValueError('vision/image budgets belong only to the coordinator')
    for name in _FIELDS:
        value=getattr(args,name,None)
        if value is not None and (type(value) is not int or not 1<=value<=(65536 if name=='max_patches' else 2048)):
            raise ValueError('invalid Flash host image budget')
    return True


def limits(args):
    return dict(max_weight_bytes=(args.vision_weight_mib or 256)*1024**2,max_tokens=args.max_patches or 4096,
        max_workspace_bytes=args.workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)


def generation_options(args):
    return dict(image_workspace_bytes=(args.image_workspace_mib or 64)*1024**2,
                image_output_bytes=(args.image_output_mib or 64)*1024**2)


def validate_metadata(args,config):
    from scripts.serve_cpu_reference import image_metadata
    image_metadata(config,args)


@dataclass(frozen=True)
class ImageRequest:
    messages: list
    images: list
    tools: object
    options: dict
    maximum: int
    prompts: list

    def generate(self,runner):
        options=dict(self.options,max_new_tokens=self.maximum)
        if self.tools is None:return runner.generate_images(self.messages,self.images,**options)
        return runner.generate_image_tools(self.messages,self.images,self.tools,**options)


def prepare_request(args,text,config):
    if args.messages is None:raise ValueError('Flash host images require an inline messages file')
    if not 1<=args.max_new_tokens<=4096 or not 1<=args.prefill_chunk<=4096:
        raise ValueError('invalid Flash host image output/chunk limit')
    tools=read_json(args.tools,1024**2) if args.tools is not None else None
    if args.tools is not None:FlashTools(tools)
    messages=read_json(args.messages,16*1024**2)
    decoded=decode_inline_images(messages,tools=tools)
    if not decoded.images:raise ValueError('Flash host image request requires at least one inline image')
    image_options=generation_options(args)
    plan=plan_images(decoded.images,max_patches=args.max_patches or 4096,
        max_workspace_bytes=image_options['image_workspace_bytes'],max_output_bytes=image_options['image_output_bytes'])
    options=dict(reasoning_effort=args.reasoning_effort,clear_thinking=args.clear_history_reasoning)
    tokens=(text.image_chat_tokens(decoded.messages,plan['grids'],**options) if tools is None
            else text.image_tool_chat_tokens(decoded.messages,plan['grids'],tools,**options))
    c=config['text_config'];output=args.output_mib*1024**2
    if (len(tokens)+args.max_new_tokens>args.context or len(tokens)*c['hidden_size']*4>output
            or min(len(tokens),args.prefill_chunk)*c['hidden_size']*c['hc_mult']*4>output):
        raise ValueError('Flash host image prompt exceeds context or prepared output budget')
    return ImageRequest(decoded.messages,decoded.images,tools,options,args.max_new_tokens,[tokens])


def load_vision(args,plan,verified):
    """Finish local vision I/O before any peer is asked to wait for commands."""
    from damendaemon.glm_flash_vision_loader import load_reference_vision
    if verified is not None:verified.verify_unchanged()
    vision=load_reference_vision(args.snapshot,**limits(args))
    if vision.source_config!=plan.config:raise ValueError('Flash host vision source differs from topology')
    if verified is not None:
        verified.verify_unchanged();vision._host_source_identity=plan.source_identity
    return vision
