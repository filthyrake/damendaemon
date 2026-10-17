"""Isolated reuse of the bounded graph cache for qualified short Q8 OLMoE."""
import hashlib
from pathlib import Path
import torch


def install_validation_dispatch():
    from damendaemon.moe import OlmoeExecutor
    from damendaemon.prefill_graphs import configuration, PrefillGraphs
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('OLMoE prefill graph experiment is limited to A2')
    original = OlmoeExecutor.configure_prefill_graphs
    report = dict(kind='experimental bounded single-prompt Q8 OLMoE prefill graphs',
                  configured=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def configure(self, use_graph):
        original(self,use_graph)
        count, tokens = configuration()
        if (use_graph and self.grouped_prefill and self.device=='cuda'
                and not hasattr(self,'pool') and count):
            self.prefill_graph_cache = PrefillGraphs(count,min(tokens,128))
            report['configured'] += 1

    OlmoeExecutor.configure_prefill_graphs = configure
    return report
