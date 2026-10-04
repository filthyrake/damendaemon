"""Bounded cross-encoder request validation and stable relevance ranking."""


def options(query,documents,top_n=None,return_documents=False):
    if not isinstance(query,str) or not query:raise ValueError('query must be nonempty text')
    if (not isinstance(documents,list) or not 1<=len(documents)<=64
            or any(not isinstance(doc,str) or not doc for doc in documents)):
        raise ValueError('documents must contain 1 to 64 nonempty texts')
    if top_n is None:top_n=len(documents)
    if type(top_n) is not int or not 1<=top_n<=len(documents):raise ValueError('invalid top_n')
    if type(return_documents) is not bool:raise ValueError('return_documents must be boolean')
    return query,list(documents),top_n,return_documents


def ranked(scores,documents,top_n,return_documents):
    import math
    if len(scores)!=len(documents) or any(not math.isfinite(value) for value in scores):
        raise RuntimeError('invalid reranker output')
    order=sorted(range(len(scores)),key=lambda index:(-scores[index],index))[:top_n]
    return [{'index':i,'relevance_score':scores[i],
             **({'document':{'text':documents[i]}} if return_documents else {})} for i in order]
