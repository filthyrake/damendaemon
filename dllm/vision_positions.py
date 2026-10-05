"""Still-image token positions for the experimental Qwen3.5 path."""
import torch


def image_positions(tokens, grids, *, image_token=248056):
    """Return three-axis prompt positions and the subsequent decode offset.

    Image placeholders must already be expanded to one token per merged feature.
    Text advances all axes together; images advance by their longest grid axis.
    Physical KV positions remain token indices and must not use this offset.
    """
    if not tokens or any(type(token) is not int or token<0 for token in tokens):
        raise ValueError('invalid image prompt tokens')
    positions=torch.empty((3,len(tokens)),dtype=torch.long)
    offset=0
    index=0
    image=0
    while index<len(tokens):
        if tokens[index]!=image_token:
            end=index+1
            while end<len(tokens) and tokens[end]!=image_token:end+=1
            positions[:,index:end]=torch.arange(offset,offset+end-index)
            offset+=end-index
        else:
            if image>=len(grids):raise ValueError('image placeholders have no matching image')
            grid=grids[image]
            if (len(grid)!=3 or any(type(v) is not int for v in grid) or grid[0]!=1
                    or min(grid[1:])<2 or grid[1]%2 or grid[2]%2):
                raise ValueError('invalid still-image grid')
            height,width=grid[1]//2,grid[2]//2
            end=index+height*width
            if end>len(tokens) or any(token!=image_token for token in tokens[index:end]):
                raise ValueError('image placeholder count does not match feature count')
            positions[0,index:end]=offset
            positions[1,index:end]=torch.arange(height).repeat_interleave(width)+offset
            positions[2,index:end]=torch.arange(width).repeat(height)+offset
            offset+=max(height,width)
            image+=1
        index=end
    if image!=len(grids):raise ValueError('image has no matching placeholders')
    return positions,offset-len(tokens)


def rotary_tables(positions, *, theta=10000000., rotary_dim=64, sections=(11,11,10)):
    if positions.ndim!=2 or positions.shape[0]!=3 or positions.dtype!=torch.long:
        raise ValueError('expected three-axis integer positions')
    if rotary_dim!=64 or tuple(sections)!=(11,11,10) or theta!=10000000.:
        raise ValueError('unsupported multimodal rotary configuration')
    frequency=1/(theta**(torch.arange(0,rotary_dim,2,device=positions.device).float()/rotary_dim))
    angles=positions.float()[:,:,None]*frequency
    combined=angles[0].clone()
    for axis in (1,2):
        indices=slice(axis,sections[axis]*3,3)
        combined[:,indices]=angles[axis,:,indices]
    combined=torch.cat((combined,combined),dim=-1)
    return combined.cos(),combined.sin()
