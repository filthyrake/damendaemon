"""Explicit FP32 attention control for diagnosing experimental OLMoE routing."""
import hashlib
from pathlib import Path
import torch


def install_validation_dispatch():
    from damendaemon.moe import OlmoeExecutor
    report = dict(kind='quadratic FP32 eager attention diagnostic; not a performance claim',
                  calls=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def attention(self, query, key, value, query_start=0):
        if query.dtype != torch.float32 or key.shape[-2] > 2048 or query.shape[1] != key.shape[1]:
            raise ValueError('unsupported OLMoE diagnostic attention shape')
        report['calls'] += 1
        scores = torch.matmul(query,key.transpose(-1,-2))*(query.shape[-1]**-.5)
        qpos = torch.arange(query.shape[-2],device=query.device)+query_start
        kpos = torch.arange(key.shape[-2],device=query.device)
        scores.masked_fill_(kpos[None,:]>qpos[:,None],torch.finfo(scores.dtype).min)
        return torch.matmul(scores.softmax(-1),value)

    OlmoeExecutor.prefill_attention = attention
    return report
