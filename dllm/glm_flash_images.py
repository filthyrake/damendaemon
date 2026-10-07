"""Pinned GLM Flash still-image CPU tensor processing; no decoding or URL I/O.

Independent implementation of Transformers v5.17.0 Glm5NextImageProcessor
(HuggingFace Team, Apache-2.0; docs/licenses/transformers-Apache-2.0.txt).
The checkpoint's processor_config.json is pinned to
zai-org/GLM-5.3-Flash eb9eb208eb0d988989d07a6a12d0fdeb5f52574a.
"""
import math
import torch
from torch.nn import functional as F

MEAN=(.48145466,.4578275,.40821073)
STD=(.26862954,.26130258,.27577711)
CONTRACT=dict(do_rescale=True,patch_expand_factor=1,merge_size=2,image_mean=list(MEAN),image_std=list(STD),
              temporal_patch_size=2,patch_size=14,min_image_tokens=16,max_image_tokens=8000,
              image_processor_type='Glm5NextImageProcessor')


def validate_processor(config):
    if not isinstance(config,dict) or set(config)!=set(CONTRACT):raise ValueError('unsupported Flash still-image processor fields')
    for name,expected in CONTRACT.items():
        value=config[name]
        if type(value) is not type(expected) or value!=expected:
            raise ValueError('unsupported Flash still-image processor option: '+name)


def image_layout(height,width):
    """Return aligned canvas and unpadded content size for the pinned still-image policy."""
    if any(type(n) is not int or not 1<=n<=2**20 for n in (height,width)):
        raise ValueError('invalid Flash image dimensions')
    factor=28;minimum=16*factor**2;maximum=8000*factor**2
    def align(n):return math.ceil(n/factor)*factor
    canvas_h,canvas_w=align(height),align(width)
    if canvas_h*canvas_w<minimum:
        scale=math.sqrt(minimum/(height*width))
        canvas_h,canvas_w=align(max(1,math.ceil(height*scale))),align(max(1,math.ceil(width*scale)))
    if canvas_h*canvas_w>maximum:
        low,high=1,height;canvas_h=canvas_w=factor
        while low<=high:
            content_h=(low+high)//2;content_w=max(1,math.floor(width*content_h/height))
            candidate_h,candidate_w=align(content_h),align(content_w)
            if candidate_h*candidate_w<=maximum:
                canvas_h,canvas_w=candidate_h,candidate_w;low=content_h+1
            else:high=content_h-1
    scale=min(canvas_h/height,canvas_w/width)
    if height*width>=minimum:scale=min(1.,scale)
    content_h=max(1,min(canvas_h,math.floor(height*scale)))
    content_w=max(1,min(canvas_w,math.floor(width*scale)))
    return canvas_h,canvas_w,content_h,content_w


@torch.inference_mode()
def prepare_images(images,*,processor_config=None,max_patches=4096,max_output_bytes=64*1024**2,
                   max_workspace_bytes=64*1024**2):
    """Return owned FP32 [patches,1176] rows and [1,rows,columns] grids.

    All images and complete retained output/scratch budgets are checked before
    resize or normalization. Caller-owned uint8 images and library allocator
    overhead are excluded. Still images duplicate the temporal patch dimension.
    """
    validate_processor(CONTRACT if processor_config is None else processor_config)
    if (not isinstance(images,(list,tuple)) or not 1<=len(images)<=256
            or type(max_patches) is not int or not 1<=max_patches<=65536
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_output_bytes,max_workspace_bytes))):
        raise ValueError('invalid Flash image list or byte budgets')
    layouts=[];count=0;workspace=0
    for image in images:
        if (not isinstance(image,torch.Tensor) or image.device.type!='cpu' or image.dtype!=torch.uint8
                or image.ndim!=3 or image.shape[2]!=3):raise ValueError('Flash images must be CPU HWC uint8 RGB tensors')
        h,w=image.shape[:2];layout=image_layout(h,w);ch,cw,_,_=layout
        layouts.append(layout);count+=ch*cw//196
        workspace=max(workspace,64*3*(h*w+ch*cw))
    if count>max_patches or 4*count*1176>max_output_bytes or workspace>max_workspace_bytes:
        raise ValueError('Flash image patch/output/workspace budget exceeded')
    output=torch.empty((count,1176),dtype=torch.float32,device='cpu');offset=0;grids=[]
    mean=torch.tensor(MEAN,dtype=torch.float32,device='cpu')[:,None,None]
    std=torch.tensor(STD,dtype=torch.float32,device='cpu')[:,None,None]
    for image,(h,w,content_h,content_w) in zip(images,layouts):
        value=image.permute(2,0,1).unsqueeze(0)
        if value.shape[-2:]!=(content_h,content_w):
            # Torchvision v0.21 CPU bicubic uses PyTorch's native uint8 kernel.
            value=F.interpolate(value,size=(content_h,content_w),mode='bicubic',align_corners=False,antialias=True)
        value=F.pad(value,(0,w-content_w,0,h-content_h))[0]
        value=(value.float()/255.-mean)/std
        rows,columns=h//14,w//14;n=rows*columns
        value=value.reshape(3,rows//2,2,14,columns//2,2,14).permute(1,4,2,5,0,3,6)
        value=value.unsqueeze(5).expand(-1,-1,-1,-1,-1,2,-1,-1)
        output[offset:offset+n]=value.reshape(n,1176);offset+=n;grids.append([1,rows,columns])
    return output,grids
