"""Isolated aggregate prefill budget for the qualified 512-context Q4 model.

Per-request context and KV limits are unchanged. This experiment does not add
an activation reserve to admission and cannot establish production support.
"""
import hashlib
from pathlib import Path

from damendaemon.batch import BatchExecutor


def install_validation_budget(budget):
    if budget not in (1024,2048):
        raise ValueError('experimental prefill budget must be 1024 or 2048')
    original = BatchExecutor.__init__
    evidence = dict(kind='isolated Q4 Marlin aggregate prefill budget',
                    requested_tokens=budget, per_request_context=512,
                    production_admission_accounts_for_larger_activations=False,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    executors=[])

    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if self.context!=512 or getattr(self,'q4_marlin_abi',None)!=2:
            return
        self.prefill_token_budget=min(budget,self.context*self.max_batch)
        evidence['executors'].append(dict(kind=type(self).__name__,context=self.context,
            max_batch=self.max_batch,prefill_token_budget=self.prefill_token_budget))
        if self.prefill_token_budget>self.context:
            self.kernel_backend+=f'+experimental-prefill-budget{self.prefill_token_budget}'

    BatchExecutor.__init__=initialize
    return evidence
