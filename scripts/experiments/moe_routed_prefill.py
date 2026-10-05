"""Use the existing routed decode projections for very small prompt batches."""
import hashlib
from pathlib import Path
import torch


def install_validation_dispatch(max_tokens=4):
    from dllm.moe import OlmoeExecutor
    if torch.cuda.get_device_name() != 'NVIDIA A2' or max_tokens not in (1,2,4,8,16):
        raise ValueError('unsupported routed prefill experiment scope')
    original = OlmoeExecutor.experts
    report = dict(kind='experimental routed Q8 expert projections for tiny prefills',
                  max_tokens=max_tokens,selected_calls=0,fallback_calls=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def experts(self, hidden, layer, prefill):
        if prefill and self.grouped_prefill and 1 <= hidden.shape[0] <= max_tokens:
            report['selected_calls'] += 1
            # Only the expert projection strategy changes; attention, positions,
            # cache writes, probabilities and model-level prefill stay intact.
            return original(self,hidden,layer,False)
        report['fallback_calls'] += 1
        return original(self,hidden,layer,prefill)

    OlmoeExecutor.experts = experts
    return report
