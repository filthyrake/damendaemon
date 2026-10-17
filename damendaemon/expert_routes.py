"""Stable expert groups with one host count vector, not a nonzero read per expert."""
import torch


def grouped_routes(selected,experts):
    if (selected.ndim!=2 or selected.dtype!=torch.int64 or not selected.shape[1]
            or type(experts) is not int or experts<1):
        raise ValueError('expert groups need an int64 route matrix and positive expert count')
    flat=selected.reshape(-1)
    order=torch.argsort(flat,stable=True)
    counts=torch.bincount(flat,minlength=experts).cpu().tolist()
    if len(counts)!=experts:raise ValueError('expert route exceeds configured expert count')
    tokens=torch.div(order,selected.shape[1],rounding_mode='floor')
    start=0
    for expert,count in enumerate(counts):
        if count:
            end=start+count
            yield expert,tokens[start:end],order[start:end]
            start=end
