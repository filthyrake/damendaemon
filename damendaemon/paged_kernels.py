"""Bulk copy between physical KV pages and contiguous prompt snapshots."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _copy(CACHE, TABLE, DENSE, SLOT: tl.constexpr, TOKENS: tl.constexpr,
          HEADS: tl.constexpr, DIM: tl.constexpr, PAGE: tl.constexpr,
          CAPACITY: tl.constexpr, TABLE_WIDTH: tl.constexpr,
          RESTORE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    valid = index < 2*HEADS*TOKENS*DIM
    column = index % DIM
    token = index // DIM % TOKENS
    head = index // (DIM*TOKENS) % HEADS
    plane = index // (DIM*TOKENS*HEADS)
    physical = tl.load(TABLE + SLOT*TABLE_WIDTH + token//PAGE,valid,other=0)
    offset = (((plane*CAPACITY+physical)*HEADS+head)*PAGE+token%PAGE)*DIM+column
    if RESTORE:
        value = tl.load(DENSE+index,valid,other=0)
        tl.store(CACHE+offset,value,valid)
    else:
        value = tl.load(CACHE+offset,valid,other=0)
        tl.store(DENSE+index,value,valid)


def copy_pages(cache,table,slot,tokens,snapshot=None):
    """Gather when snapshot is absent; otherwise restore exactly its live tokens.

    The executor owns page admission and must allocate all destination pages
    before calling. Unused entries and the tail of the last page are untouched.
    """
    if (cache.ndim!=5 or cache.shape[0]!=2 or not cache.is_contiguous()
            or type(slot) is not int or not 0<=slot<table.shape[0]
            or type(tokens) is not int or not 0<tokens<=table.shape[1]*cache.shape[-2]):
        raise ValueError('invalid paged copy shape')
    shape=(2,cache.shape[2],tokens,cache.shape[4])
    restore=snapshot is not None
    if restore:
        if snapshot.shape!=shape or snapshot.dtype!=cache.dtype or snapshot.device!=cache.device or not snapshot.is_contiguous():
            raise ValueError('invalid KV snapshot')
    else:
        snapshot=torch.empty(shape,dtype=cache.dtype,device=cache.device)
    _copy[(tr.cdiv(snapshot.numel(),1024),)](cache,table,snapshot,slot,tokens,cache.shape[2],cache.shape[4],
        cache.shape[3],cache.shape[1],table.shape[1],restore,1024)
    return snapshot
