"""Immutable bounded contracts for explicit CPU host pipeline stages."""
from dataclasses import dataclass
import hashlib
import json

from .cpu_hybrid_stage import cpu_hybrid_shapes
from .host_hybrid_worker import _token_policy
from .host_tensor_channel import MAX_FRAME_BYTES
from .stage_protocol import StageState

VERSION=1


@dataclass(frozen=True,init=False)
class HostTopology:
    config_json: bytes
    ranges: tuple
    source_identity: bytes
    session_identity: bytes | None
    quantization: str
    input_kind: str
    context: int
    context_limit: int
    slots: int
    chunk_size: int
    frame_bytes: int
    width: int
    mapped_vocab_size: int
    excluded_ids: tuple
    identity: bytes

    def __init__(self,config,ranges,source_identity,*,quantization='none',context=512,context_limit=512,
                 slots=4,chunk_size=128,frame_bytes=MAX_FRAME_BYTES,mapped_vocab_size=None,excluded_ids=(),session_identity=None,input_kind='tokens'):
        if not isinstance(config,dict):raise ValueError('host topology requires prepared CPU configuration')
        try:encoded=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        except (TypeError,ValueError,RecursionError) as error:raise ValueError('invalid topology configuration metadata') from error
        if len(encoded)>1024**2:raise ValueError('topology configuration exceeds 1 MiB')
        config=json.loads(encoded)
        glm=config.get('model_type')=='glm_moe_dsa'
        flash=config.get('model_type')=='glm5_next'
        deepseek=config.get('model_type')=='deepseek_v4'
        if input_kind not in ('tokens','flash_embeddings') or (input_kind!='tokens' and not flash):
            raise ValueError('prepared embedding input requires an explicit Flash topology')
        text=config['text_config'] if flash and isinstance(config.get('text_config'),dict) else config
        if glm:
            from .glm_checkpoint import tensor_specs
            from .glm_reference_loader import _options
            tensor_specs(config);_options(config)
        elif flash:
            from .glm_flash_checkpoint import tensor_specs
            from .glm_flash_reference_loader import _options
            tensor_specs(config);_options(text)
        elif deepseek:
            from .deepseek_checkpoint import tensor_specs
            from .deepseek_reference_loader import _options
            from .deepseek_layer_stage import validate_range
            tensor_specs(config);_options(config);validate_range(0,config['num_hidden_layers'],config['num_hidden_layers'])
        else:cpu_hybrid_shapes(config,0,config.get('num_hidden_layers'))
        width=text['hidden_size']*(text['hc_mult'] if flash or deepseek else 1)
        if not isinstance(source_identity,bytes) or len(source_identity)!=32:raise ValueError('topology needs a 32-byte source/session identity')
        if session_identity is not None and (not isinstance(session_identity,bytes) or len(session_identity)!=32):
            raise ValueError('optional topology session identity must contain 32 bytes')
        if quantization not in (('fp4_fp8_reference',) if deepseek else ('fp8_reference',) if glm or flash else ('none','q4_0','q8_0')):raise ValueError('unsupported CPU topology quantization')
        if not isinstance(ranges,(list,tuple)) or not 2<=len(ranges)<=8:raise ValueError('host topology needs 2..8 stages')
        end=0;normalized=[]
        for pair in ranges:
            if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(n) is not int for n in pair)
                    or pair[0]!=end or not end<pair[1]<=text['num_hidden_layers']):
                raise ValueError('host stages must be contiguous nonempty ordered layer ranges')
            if glm:
                from .glm_layer_stage import validate_range
                validate_range(*pair,text['num_hidden_layers'],config['indexer_types'])
            normalized.append(tuple(pair));end=pair[1]
        if end!=text['num_hidden_layers']:raise ValueError('host stages must cover every model layer')
        StageState(1,context,slots,context_limit=context_limit)
        limit=text.get('max_position_embeddings',2**20)
        if type(limit) is not int or not 1<=limit<=2**31 or context>limit:
            raise ValueError('topology context exceeds valid model positions')
        if (type(chunk_size) is not int or not 1<=chunk_size<=4096 or type(frame_bytes) is not int
                or not 256<=frame_bytes<=MAX_FRAME_BYTES
                or max(slots,min(context,chunk_size))*max(width*4,8 if deepseek else 0)>frame_bytes):
            raise ValueError('topology activation chunk/decode exceeds frame budget')
        policy_config=dict(text)
        if flash:
            media=text.get('multimodal_token_ids',[])
            if not isinstance(media,(list,tuple)):
                raise ValueError('Flash topology media exclusions must be a bounded sequence')
            media=list(media)
            media.extend(config[key] for key in ('image_token_id','video_token_id','audio_token_id',
                'image_start_token_id','image_end_token_id','video_start_token_id','video_end_token_id',
                'audio_start_token_id','audio_end_token_id') if key in config)
            policy_config['multimodal_token_ids']=media
        mapped,excluded=_token_policy(policy_config,mapped_vocab_size,excluded_ids)
        manifest=dict(version=VERSION,backend='cpu-float32',config=config,ranges=normalized,
            source_identity=source_identity.hex(),quantization=quantization,context=context,context_limit=context_limit,
            slots=slots,chunk_size=chunk_size,frame_bytes=frame_bytes,
            mapped_vocab_size=mapped,excluded_ids=sorted(excluded))
        if input_kind!='tokens':manifest['input_kind']=input_kind
        if deepseek:manifest['routing_tokens']='int64-before-activation-v1'
        if session_identity is not None:manifest['session_identity']=session_identity.hex()
        identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
        values=dict(config_json=encoded,ranges=tuple(normalized),source_identity=source_identity,session_identity=session_identity,quantization=quantization,input_kind=input_kind,
            context=context,context_limit=context_limit,slots=slots,chunk_size=chunk_size,frame_bytes=frame_bytes,width=width,
            mapped_vocab_size=mapped,excluded_ids=tuple(sorted(excluded)),identity=identity)
        for name,value in values.items():object.__setattr__(self,name,value)

    @property
    def config(self):return json.loads(self.config_json)

    @property
    def routing_tokens(self):return self.config.get('model_type')=='deepseek_v4'

    @property
    def text_config(self):
        config=self.config
        return config['text_config'] if config.get('model_type')=='glm5_next' else config

    def result_kind(self,index):
        if type(index) is not int or not 0<=index<len(self.ranges):raise ValueError('invalid global stage index')
        return 'tokens' if index==len(self.ranges)-1 else 'activation'

    def stage_identity(self,index):
        kind=self.result_kind(index)
        return hashlib.sha256(b'damendaemon-host-stage-v1\0'+self.identity+bytes([index])+kind.encode()).digest()
