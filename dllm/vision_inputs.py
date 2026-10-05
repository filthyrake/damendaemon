"""Bounded RGB tensor preprocessing for the experimental Qwen3.5 encoder."""
import math

import torch
from torch.nn import functional as F


def image_size(height, width, *, min_pixels=65536, max_pixels=16777216):
    if (type(height) is not int or type(width) is not int or min(height,width)<1
            or max(height,width)>200*min(height,width)):
        raise ValueError('invalid image dimensions or aspect ratio')
    if (type(min_pixels) is not int or type(max_pixels) is not int
            or min_pixels<1024 or max_pixels<min_pixels):
        raise ValueError('invalid image area limits')
    factor=32
    target=[round(size/factor)*factor for size in (height,width)]
    area=target[0]*target[1]
    if area>max_pixels:
        scale=math.sqrt(height*width/max_pixels)
        target=[max(factor,math.floor(size/scale/factor)*factor) for size in (height,width)]
    elif area<min_pixels:
        scale=math.sqrt(min_pixels/(height*width))
        target=[math.ceil(size*scale/factor)*factor for size in (height,width)]
    return tuple(target)


def prepare_images(images, *, max_patches=4096):
    """Convert CPU HWC uint8 RGB tensors to packed FP32 patches and image grids.

    Validate the complete patch budget before resizing or constructing patches.
    Encoded-file decoding and HTTP input policy belong to the serving layer.
    """
    if not isinstance(images,(list,tuple)) or not images:
        raise ValueError('at least one RGB image is required')
    if type(max_patches) is not int or max_patches<4:
        raise ValueError('invalid image patch budget')
    shapes=[]
    for image in images:
        if (not isinstance(image,torch.Tensor) or image.device.type!='cpu'
                or image.dtype!=torch.uint8 or image.ndim!=3 or image.shape[-1]!=3):
            raise ValueError('expected CPU HWC uint8 RGB image')
        shapes.append(image_size(image.shape[0],image.shape[1]))
    if sum(h*w//256 for h,w in shapes)>max_patches:
        raise ValueError('vision patch budget exceeded')
    patches=[]
    grids=[]
    for image,(height,width) in zip(images,shapes):
        value=image.permute(2,0,1).unsqueeze(0)
        if value.shape[-2:]!=(height,width):
            value=F.interpolate(value,size=(height,width),mode='bicubic',
                                align_corners=False,antialias=True)
        value=(value.float()-127.5)/127.5
        rows,columns=height//16,width//16
        value=value.reshape(3,rows//2,2,16,columns//2,2,16)
        value=value.permute(1,4,2,5,0,3,6)
        value=value.unsqueeze(5).expand(-1,-1,-1,-1,-1,2,-1,-1)
        patches.append(value.reshape(rows*columns,1536))
        grids.append([1,rows,columns])
    return torch.cat(patches),grids
