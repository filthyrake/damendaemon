"""Diagnostic image-serving stage timings; synchronization changes performance.

Run only through verify_vision_service --stage-profile. This is not imported by
normal serving and its measurements must not be used as production benchmarks.
"""
import functools
import gc
import json
import os
import time
import torch
from dllm import vision_model
from dllm.vision_hybrid import VisionHybridExecutor
from dllm.vision_qwen35 import VisionEncoder
from dllm.cli import main

output=open(os.environ['DLLM_VISION_STAGE_PROFILE'],'w',buffering=1)


def record(stage,started,**extra):
    output.write(json.dumps(dict(stage=stage,start=started,seconds=time.perf_counter()-started,**extra))+'\n')


def instrument(owner,name,*,cuda=False):
    original=getattr(owner,name)
    @functools.wraps(original)
    def measured(*args,**kwargs):
        if cuda:torch.cuda.synchronize()
        started=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:
            if cuda:torch.cuda.synchronize()
            record(name,started)
    setattr(owner,name,measured)


gc_start={}
def collection(phase,info):
    generation=info['generation']
    if phase=='start':gc_start[generation]=time.perf_counter()
    else:record('gc',gc_start.pop(generation),generation=generation,collected=info['collected'])


gc.callbacks.append(collection)
instrument(vision_model,'decode_image_messages')
instrument(vision_model,'prepare_images')
instrument(vision_model.VisionModelWeights,'chat_tokens')
instrument(VisionEncoder,'encode',cuda=True)
instrument(VisionHybridExecutor,'prefill_images',cuda=True)
main()
