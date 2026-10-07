"""Explicit prepared Flash image-feature insertion, independent of media chat APIs."""
import torch
from torch.nn import functional as F
from .glm_flash_model import FlashModel


@torch.no_grad()
def image_embeddings(model,tokens,features,*,image_token_id):
    """Replace each row's image placeholders, left-to-right, with owned vision rows.

    `features` contains one CPU FP32 [image tokens,width] tensor per batch row;
    concatenate that row's images in prompt order before calling. Zero-image
    rows provide an empty matrix. This validates count/width, not semantic image
    identity, processor provenance or chat marker syntax. No input is mutated.
    """
    if not isinstance(model,FlashModel):raise ValueError('image insertion requires a Flash reference model')
    model._tokens(tokens,False)
    if (type(image_token_id) is not int or not 0<=image_token_id<model.embedding.shape[0]
            or not isinstance(features,(list,tuple)) or len(features)!=tokens.shape[0]
            or tokens.shape[0]>128):raise ValueError('invalid Flash image token or feature batch')
    width=model.embedding.shape[1]
    for row,values in zip(tokens,features):
        if (not isinstance(values,torch.Tensor) or values.device.type!='cpu' or values.dtype!=torch.float32
                or values.ndim!=2 or values.shape[1]!=width or values.shape[0]>tokens.shape[1]
                or values.shape[0]!=int((row==image_token_id).sum()) or not torch.isfinite(values).all()):
            raise ValueError('Flash image feature count/width/storage mismatch')
    result=F.embedding(tokens,model.embedding)
    for row,ids,values in zip(result,tokens,features):row[ids==image_token_id]=values
    return result
