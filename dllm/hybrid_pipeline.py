"""Synchronous in-process hybrid stage coordinator, not a serving backend.

No remote transport, asynchronous overlap, or production multi-GPU admission.
Cursors commit only after every stage and its device synchronization succeed.
"""
import threading

import torch


class PipelineCancelled(RuntimeError):
    pass


class HybridPipeline:
    supports_chunked_prefill = True
    supports_prefix_cache = False
    requires_full_prefill = False
    kernel_backend = 'torch-hybrid-pipeline'
    residual_dtype = torch.float32

    def __init__(self, stages):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError('hybrid pipeline needs at least one stage')
        first = self.stages[0]
        end = 0
        for stage in self.stages:
            if (stage.closed or getattr(stage, '_pipeline_owner', None) is not None
                    or stage.layer_start != end or stage.layer_end <= end
                    or stage.source_identity is not first.source_identity
                    or stage.c != first.c or stage.context != first.context
                    or stage.max_batch != first.max_batch or stage.compute_dtype != first.compute_dtype
                    or any(stage.lengths)):
                raise ValueError('hybrid stages must be fresh, contiguous and share model, precision and limits')
            end = stage.layer_end
        if end != first.c['num_hidden_layers']:
            raise ValueError('hybrid stages must cover the entire model')
        self.owner = object()
        for stage in self.stages:
            stage._pipeline_owner = self.owner
        self.context, self.max_batch = first.context, first.max_batch
        self.compute_dtype = first.compute_dtype
        self.lengths = [0]*self.max_batch
        self.failed = self.closed = False
        self.lock = threading.RLock()

    def _ready(self):
        if self.closed or self.failed:
            raise RuntimeError('hybrid pipeline is closed or failed')

    def _tokens(self, tokens):
        c = self.stages[0].c
        if (not isinstance(tokens, (list, tuple)) or not tokens
                or any(type(t) is not int or not 0 <= t < c['vocab_size'] for t in tokens)
                or set(tokens) & set(c.get('multimodal_token_ids', []))):
            raise ValueError('invalid hybrid text tokens')

    @torch.inference_mode()
    def _execute(self, rows, slots, starts, prefill, final, cancelled):
        if cancelled is not None and cancelled():
            raise PipelineCancelled('cancelled before stage execution')
        touched = False
        try:
            hidden = None
            for stage in self.stages:
                if cancelled is not None and cancelled():
                    raise PipelineCancelled('cancelled during stage execution')
                touched = True
                device = stage.actual_device
                positions = torch.tensor([[start+i for i in range(len(row))]
                                          for start, row in zip(starts, rows)], device=device)
                slot_ids = torch.tensor(slots, device=device)
                if hidden is None:
                    hidden = stage.embedding(torch.tensor(rows, device=device))
                else:
                    hidden = hidden.to(device=device, non_blocking=False)
                hidden = stage.forward_layers(hidden, positions, slot_ids, prefill,
                                              starts[0] if prefill else 0)
            logits = self.stages[-1].output_logits(hidden[:, -1]) if final else None
            next_ids = logits.argmax(-1) if not prefill else None
            for stage in self.stages:
                stage.synchronize()
            if cancelled is not None and cancelled():
                raise PipelineCancelled('cancelled before cursor commit')
            # Nothing below this point launches device work.
            for slot, start, row in zip(slots, starts, rows):
                self.lengths[slot] = start+len(row)
            for stage in self.stages:
                stage.lengths[:] = self.lengths
                stage.length = self.lengths[0]
            return logits if prefill else (logits, next_ids)
        except BaseException:
            # Even a cancelled call can have mutated earlier-stage state.
            if touched:
                self.failed = True
            raise

    def prefill_chunk(self, tokens, slot=0, start=0, final=True, cancelled=None):
        with self.lock:
            self._ready()
            self._tokens(tokens)
            if (type(slot) is not int or not 0 <= slot < self.max_batch
                    or type(start) is not int or start < 0 or type(final) is not bool
                    or start+len(tokens) > self.context or (start and start != self.lengths[slot])):
                raise ValueError('invalid pipeline prompt offset or slot')
            return self._execute([list(tokens)], [slot], [start], True, final, cancelled)

    def prefill(self, tokens, slot=0, cancelled=None):
        return self.prefill_chunk(tokens, slot, cancelled=cancelled)

    def prefill_batch(self, prompts, slots):
        with self.lock:
            self._ready()
            if (not isinstance(prompts, (list, tuple)) or not prompts
                    or not isinstance(slots, (list, tuple)) or len(prompts) != len(slots)
                    or len(slots) > self.max_batch
                    or any(type(s) is not int or not 0 <= s < self.max_batch for s in slots)
                    or len(set(slots)) != len(slots)):
                raise ValueError('invalid pipeline prompt batch or slots')
            for prompt in prompts:
                self._tokens(prompt)
                if len(prompt) > self.context:
                    raise ValueError('pipeline prompt exceeds context')
            try:
                return torch.cat([self.prefill(prompt, slot) for prompt, slot in zip(prompts, slots)])
            except BaseException:
                self.failed = True
                raise

    def decode(self, token):
        return self.decode_batch([token], [0])

    def decode_batch(self, tokens, slots, cancelled=None):
        with self.lock:
            self._ready()
            self._tokens(tokens)
            if (not isinstance(slots, (list, tuple)) or len(tokens) != len(slots)
                    or len(slots) > self.max_batch
                    or any(type(s) is not int or not 0 <= s < self.max_batch for s in slots)
                    or len(set(slots)) != len(slots)
                    or any(not 0 < self.lengths[s] < self.context for s in slots)):
                raise ValueError('invalid pipeline decode slots or context')
            return self._execute([[t] for t in tokens], list(slots),
                                 [self.lengths[s] for s in slots], False, True, cancelled)

    def status(self):
        with self.lock:
            return dict(failed=self.failed, closed=self.closed, lengths=list(self.lengths),
                        stages=[dict(layer_start=s.layer_start, layer_end=s.layer_end,
                            closed=s.closed, required_memory_bytes=s.required_memory_bytes,
                            activation_workspace_bytes=s.stage_activation_workspace_bytes,
                            output_head_workspace_bytes=s.output_head_workspace_bytes,
                            resident_weight_bytes=sum(t.numel()*t.element_size()
                                for t in {id(t):t for t in s.w.values()}.values()),
                            state_bytes=sum(t.numel()*t.element_size()
                                for group in (s.kv, s.conv, s.state) for t in group.values()))
                            for s in self.stages])

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.failed = True
            # Check every device before releasing any stage's owned buffers.
            for stage in self.stages:
                stage.synchronize()
            for stage in self.stages:
                stage.close()
            self.closed = True
