"""Bounded inline image decoding for an experimental chat input path.

No network fetches or filesystem paths are accepted. This helper does not enable
multimodal serving; callers still need a qualified image-capable executor.
"""
import base64
import binascii
import copy
import hashlib
import io
import warnings
from dataclasses import dataclass

import torch

from .tool_calls import validate_messages


@dataclass
class ImageMessages:
    messages: list
    images: tuple
    identities: tuple


def _parse_image_messages(messages, *, max_images=4, max_bytes=8*1024*1024, max_pixels=16*1024*1024):
    if any(type(value) is not int or value<1 for value in (max_images,max_bytes,max_pixels)):
        raise ValueError('invalid image input limits')
    if not isinstance(messages,list) or not messages:
        raise ValueError('messages must be a nonempty list')
    normalized=[]
    encoded=[]
    total_bytes=0
    marker='<|vision_start|><|image_pad|><|vision_end|>'
    for message in messages:
        if not isinstance(message,dict):raise ValueError('invalid chat message')
        content=message.get('content')
        if not isinstance(content,list):
            normalized.append(dict(message))
            continue
        if message.get('role')!='user' or not content:
            raise ValueError('image content lists require a nonempty user message')
        text=[]
        for part in content:
            if not isinstance(part,dict):raise ValueError('invalid image content part')
            if part.get('type')=='text' and set(part)=={'type','text'} and isinstance(part['text'],str):
                text.append(part['text'])
                continue
            if part.get('type')!='image_url' or set(part)!={'type','image_url'}:
                raise ValueError('unsupported image content part')
            spec=part['image_url']
            if (not isinstance(spec,dict) or set(spec)-{'url','detail'} or not isinstance(spec.get('url'),str)
                    or spec.get('detail','auto')!='auto'):
                raise ValueError('image_url requires an inline URL and automatic detail')
            if len(encoded)>=max_images:raise ValueError('image count limit exceeded')
            url=spec['url']
            formats={'data:image/png;base64,':'PNG','data:image/jpeg;base64,':'JPEG'}
            prefix=next((prefix for prefix in formats if url.startswith(prefix)),None)
            if prefix is None:raise ValueError('only inline PNG and JPEG images are supported')
            payload=url[len(prefix):]
            if len(payload)>4*((max_bytes-total_bytes+2)//3):
                raise ValueError('image byte budget exceeded')
            try:data=base64.b64decode(payload,validate=True)
            except (ValueError,binascii.Error) as error:raise ValueError('invalid image base64') from error
            total_bytes+=len(data)
            if not data or total_bytes>max_bytes:raise ValueError('image byte budget exceeded')
            encoded.append((data,formats[prefix]))
            text.append(marker)
        normalized.append({**message,'content':''.join(text)})
    return validate_messages(normalized),encoded


def validate_image_messages(messages):
    """Validate bounded content without decoding pixels on a request thread."""
    _parse_image_messages(messages)
    return copy.deepcopy(messages)


def decode_image_messages(messages, *, max_images=4, max_bytes=8*1024*1024, max_pixels=16*1024*1024):
    normalized,encoded=_parse_image_messages(messages,max_images=max_images,max_bytes=max_bytes,max_pixels=max_pixels)
    if not encoded:return ImageMessages(normalized,(),())
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ValueError('image input requires the Pillow dependency') from error
    images=[]
    opened=[]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            area=0
            for data,format_name in encoded:
                image=Image.open(io.BytesIO(data))
                opened.append(image)
                if image.format!=format_name or getattr(image,'n_frames',1)!=1:
                    raise ValueError('image format mismatch or animated image')
                area+=image.width*image.height
                if area>max_pixels:raise ValueError('decoded image pixel budget exceeded')
            for image in opened:
                rgb=ImageOps.exif_transpose(image).convert('RGB')
                images.append(torch.frombuffer(bytearray(rgb.tobytes()),dtype=torch.uint8)
                              .reshape(rgb.height,rgb.width,3))
    except (OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ValueError('invalid or oversized encoded image') from error
    finally:
        for image in opened:image.close()
    return ImageMessages(normalized,tuple(images),tuple(hashlib.sha256(data).hexdigest() for data,_ in encoded))
