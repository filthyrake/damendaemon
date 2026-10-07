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
                 slots=4,chunk_size=128,frame_bytes=MAX_FRAME_BYTES,mapped_vocab_size=None,excluded_ids=(),session_identity=None):
        if not isinstance(config,dict):raise ValueError('host topology requires prepared CPU configuration')
        try:encoded=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        except (TypeError,ValueError,RecursionError) as error:raise ValueError('invalid topology configuration metadata') from error
        if len(encoded)>1024**2:raise ValueError('topology configuration exceeds 1 MiB')
        config=json.loads(encoded)
        glm=config.get('model_type')=='glm_moe_dsa'
        if glm:
            from .glm_checkpoint import tensor_specs
            from .glm_reference_loader import _options
            tensor_specs(config);_options(config)
        else:cpu_hybrid_shapes(config,0,config.get('num_hidden_layers'))
        if not isinstance(source_identity,bytes) or len(source_identity)!=32:raise ValueError('topology needs a 32-byte source/session identity')
        if session_identity is not None and (not isinstance(session_identity,bytes) or len(session_identity)!=32):
            raise ValueError('optional topology session identity must contain 32 bytes')
        if quantization not in (('fp8_reference',) if glm else ('none','q4_0','q8_0')):raise ValueError('unsupported CPU topology quantization')
        if not isinstance(ranges,(list,tuple)) or not 2<=len(ranges)<=8:raise ValueError('host topology needs 2..8 stages')
        end=0;normalized=[]
        for pair in ranges:
            if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(n) is not int for n in pair)
                    or pair[0]!=end or not end<pair[1]<=config['num_hidden_layers']):
                raise ValueError('host stages must be contiguous nonempty ordered layer ranges')
            if glm:
                from .glm_layer_stage import validate_range
                validate_range(*pair,config['num_hidden_layers'],config['indexer_types'])
            normalized.append(tuple(pair));end=pair[1]
        if end!=config['num_hidden_layers']:raise ValueError('host stages must cover every model layer')
        StageState(1,context,slots,context_limit=context_limit)
        limit=config.get('max_position_embeddings',2**20)
        if type(limit) is not int or not 1<=limit<=2**31 or context>limit:
            raise ValueError('topology context exceeds valid model positions')
        if (type(chunk_size) is not int or not 1<=chunk_size<=4096 or type(frame_bytes) is not int
                or not 256<=frame_bytes<=MAX_FRAME_BYTES
                or max(slots,min(context,chunk_size))*config['hidden_size']*4>frame_bytes):
            raise ValueError('topology activation chunk/decode exceeds frame budget')
        mapped,excluded=_token_policy(config,mapped_vocab_size,excluded_ids)
        manifest=dict(version=VERSION,backend='cpu-float32',config=config,ranges=normalized,
            source_identity=source_identity.hex(),quantization=quantization,context=context,context_limit=context_limit,
            slots=slots,chunk_size=chunk_size,frame_bytes=frame_bytes,
            mapped_vocab_size=mapped,excluded_ids=sorted(excluded))
        if session_identity is not None:manifest['session_identity']=session_identity.hex()
        identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
        values=dict(config_json=encoded,ranges=tuple(normalized),source_identity=source_identity,session_identity=session_identity,quantization=quantization,
            context=context,context_limit=context_limit,slots=slots,chunk_size=chunk_size,frame_bytes=frame_bytes,width=config['hidden_size'],
            mapped_vocab_size=mapped,excluded_ids=tuple(sorted(excluded)),identity=identity)
        for name,value in values.items():object.__setattr__(self,name,value)

    @property
    def config(self):return json.loads(self.config_json)

    def result_kind(self,index):
        if type(index) is not int or not 0<=index<len(self.ranges):raise ValueError('invalid global stage index')
        return 'tokens' if index==len(self.ranges)-1 else 'activation'

    def stage_identity(self,index):
        kind=self.result_kind(index)
        return hashlib.sha256(b'dllm-host-stage-v1\0'+self.identity+bytes([index])+kind.encode()).digest()
