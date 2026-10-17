"""FP64 normalization control for sensitivity diagnosis; never normal dispatch."""
import hashlib
from pathlib import Path
import torch


def install_validation_dispatch():
    from damendaemon.moe import OlmoeExecutor
    report = dict(kind='FP64 normalization diagnostic; FP32 output and remaining arithmetic',
                  calls=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def norm(self, x, weight):
        report['calls'] += 1
        value = x.double()
        normalized = value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.eps)
        return normalized.float()*weight.float()

    OlmoeExecutor.norm = norm
    return report
