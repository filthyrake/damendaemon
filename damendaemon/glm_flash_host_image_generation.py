"""Explicit native still-image/tool policy over owned prepared Flash host stages."""
from contextlib import closing
import hashlib
import json
import torch
from torch.nn import functional as F
from .glm_flash_host_generation import HostFlashGenerator,_policy as text_policy
from .glm_flash_image_tool_text import FlashImageToolText,_render_image_tool_chat
from .glm_flash_image_text import BEGIN_IMAGE,IMAGE,END_IMAGE,_render
from .glm_flash_images import CONTRACT,plan_images,prepare_images
from .glm_flash_tools import FlashTools,_json,_dump
from .glm_flash_vision import FlashVision
from .cuda_host_stage import CudaHostStageSlots
from .native_host_generation import snapshot_text
from .reference_cancellation import check_cancelled


def _snapshot(text):
    result=snapshot_text(text,FlashImageToolText)
    ids=[]
    for marker in (BEGIN_IMAGE,IMAGE,END_IMAGE):
        token=result.tokenizer.token_to_id(marker)
        if token is None or result.tokenizer.encode(marker,add_special_tokens=False).ids!=[token]:
            raise ValueError('native host images require atomic image markers')
        ids.append(token)
    if len(set(ids))!=3 or getattr(text,'image_token_id',None)!=ids[1]:
        raise ValueError('native host image token policy disagrees')
    result.image_token_id=ids[1]
    return result


def _policy(text,verified):
    manifest=dict(version='damendaemon-flash-host-images-v1',text_policy=text_policy(text,verified).hex(),
        processor=CONTRACT,image_token_id=text.image_token_id,
        image_markers=[text.tokenizer.token_to_id(marker) for marker in (BEGIN_IMAGE,IMAGE,END_IMAGE)])
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()


def native_image_session_identity(text,*,verified=False):return _policy(_snapshot(text),verified)


def load_host_image_vision(path,worker,*,identity_max_bytes=2*1024**3,identity_timeout=60,**limits):
    """Load coordinator vision; verified sessions recheck the full snapshot bytes.

    The returned vision parameters are borrowed immutably by the generator.
    The fixed processor contract is bound by native_image_session_identity.
    """
    from .cpu_snapshot_identity import fingerprint
    from .glm_flash_vision_loader import load_reference_vision
    from .host_pipeline_worker import HostPipelineCoordinator
    if not isinstance(worker,HostPipelineCoordinator) or worker.topology.input_kind!='flash_embeddings':
        raise ValueError('host image vision requires prepared Flash coordinator')
    worker._ready(True)
    identity=None
    if getattr(worker,'model_identity_verified',False):
        identity=fingerprint(path,max_bytes=identity_max_bytes,timeout=identity_timeout)
        if bytes.fromhex(identity.model_id)!=worker.topology.source_identity:
            raise ValueError('host vision snapshot differs from verified language source')
    vision=load_reference_vision(path,**limits)
    if vision.source_config!=worker.topology.config:raise ValueError('host vision configuration differs from language stages')
    if identity is not None:
        identity.verify_unchanged();vision._host_source_identity=worker.topology.source_identity
    return vision


class HostFlashImageGenerator(HostFlashGenerator):
    """Native single-image conversation owner; prepared immutable CPU vision.

    The image list may contain multiple still images in prompt order. Ordinary
    native text/tool methods remain available. Command/HTTP admission is separate.
    """
    text_type=FlashImageToolText
    input_kind='flash_embeddings'
    policy=staticmethod(_policy)

    def _snapshot_text(self,text):return _snapshot(text)

    def __init__(self,worker,text,vision,*,chunk_size=128,
                 image_workspace_bytes=64*1024**2,image_output_bytes=64*1024**2):
        from .host_pipeline_worker import HostPipelineCoordinator
        if not isinstance(worker,HostPipelineCoordinator):raise ValueError('native images require a prepared coordinator')
        config=worker.topology.config;vc=config.get('vision_config',{})
        if not isinstance(vc,dict):raise ValueError('native host images require vision configuration')
        expected=dict(vc);expected['rope_parameters']=dict(vc.get('rope_parameters') or {})
        if (type(vision) is not FlashVision or vision.c!=expected
                or vision.c.get('out_hidden_size')!=worker.topology.text_config['hidden_size']
                or tuple(vision.c.get(k) for k in ('in_channels','patch_size','temporal_patch_size','spatial_merge_size'))!=(3,14,2,2)
                or getattr(vision,'source_config',config)!=config
                or config.get('image_token_id')!=getattr(text,'image_token_id',None)
                or any(type(v) is not int or not 1<=v<=2**31 for v in (image_workspace_bytes,image_output_bytes))):
            raise ValueError('native host image vision/language geometry or budgets disagree')
        if (getattr(worker,'model_identity_verified',False)
                and getattr(vision,'_host_source_identity',None)!=worker.topology.source_identity):
            raise ValueError('verified native images require vision loaded from the same verified snapshot')
        if (not isinstance(worker.stage,CudaHostStageSlots)
                and worker.stage.stage.embedding.device.type!='cpu'):
            raise ValueError('host image embedding requires CPU weights or an explicit CUDA boundary')
        # Claim the ready coordinator only after all image-specific checks.
        super().__init__(worker,text,chunk_size=chunk_size)
        self.vision=vision;self.image_workspace_bytes=image_workspace_bytes;self.image_output_bytes=image_output_bytes
        self.feature_output_bytes=worker.stage.stage.max_output_bytes

    def close(self):
        super().close()
        self.vision=None

    def _text_embeddings(self,ids):
        if isinstance(self.worker.stage,CudaHostStageSlots):
            try:return self.worker.stage.text_embeddings(ids[None])[0]
            except BaseException as error:
                # This preparation precedes protocol dispatch. A failed copy
                # retires its CUDA owner, so retire the host epoch too; ordinary
                # admission rejection leaves the untouched epoch reusable.
                if self.worker.stage.failed or self.worker.stage.closed:self.worker._fail(error)
                raise
        return F.embedding(ids,self.worker.stage.stage.embedding)

    def _prepare_images(self,messages,images,maximum,tools,options):
        if type(maximum) is not int or not 1<=maximum<=4096:raise ValueError('native image output length must be 1..4096')
        codec=None
        if tools is None:prompt,count=_render(messages,**options)
        else:
            declarations=_json(_dump(tools));codec=FlashTools(declarations)
            prompt,count=_render_image_tool_chat(messages,declarations,**options)
        if not isinstance(images,(list,tuple)) or not 1<=len(images)==count:
            raise ValueError('native image response requires one RGB tensor per marker')
        self.completion._check_cancelled(None);check_cancelled()
        plan=plan_images(images,max_patches=self.vision.max_tokens,max_workspace_bytes=self.image_workspace_bytes,
            max_output_bytes=self.image_output_bytes)
        tokens=self.text._expand_image_tokens(prompt,count,plan['grids'])
        if (not tokens or len(tokens)+maximum>self.max_context
                or any(type(token) is not int or not 0<=token<self.text.tokenizer_size for token in tokens)
                or len(tokens)*self.worker.topology.text_config['hidden_size']*4>self.feature_output_bytes
                or min(len(tokens),self.chunk_size)*self.worker.width*4>min(self.feature_output_bytes,self.worker.topology.frame_bytes)):
            raise ValueError('native image prompt exceeds context or prepared feature bounds')
        # Processor input is bounded by plan_images; own pixels before yielding.
        images=[image.clone() for image in images]
        def prepare():
            check_cancelled()
            patches,grids=prepare_images(images,max_patches=self.vision.max_tokens,
                max_workspace_bytes=self.image_workspace_bytes,max_output_bytes=self.image_output_bytes)
            features=self.vision.forward(patches,grids)
            del patches
            check_cancelled();self.completion._check_cancelled(None)
            ids=torch.tensor(tokens,dtype=torch.int64)
            if (not isinstance(features,torch.Tensor) or features.dtype!=torch.float32 or features.device.type!='cpu' or features.ndim!=2
                    or tuple(features.shape)!=(int((ids==self.text.image_token_id).sum()),self.worker.topology.text_config['hidden_size'])
                    or not torch.isfinite(features).all()):raise ValueError('invalid native host image feature projection')
            with torch.inference_mode():
                values=self._text_embeddings(ids)
                values[ids==self.text.image_token_id]=features
            return [values]
        return [tokens],[maximum],codec,prepare

    def generate_images(self,messages,images,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prompts,limits,codec,prepare=self._prepare_images(messages,images,max_new_tokens,None,options)
        with closing(self._generate(prompts,limits,codec,prepare_embeddings=prepare)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}

    def generate_image_tools(self,messages,images,tools,*,max_new_tokens=128,reasoning_effort=None,clear_thinking=False):
        if tools is None:raise ValueError('native image tools require function declarations')
        options=dict(reasoning_effort=reasoning_effort,clear_thinking=clear_thinking)
        prompts,limits,codec,prepare=self._prepare_images(messages,images,max_new_tokens,tools,options)
        with closing(self._generate(prompts,limits,codec,prepare_embeddings=prepare)) as events:
            for event in events:yield {key:value for key,value in event.items() if key!='row'}
