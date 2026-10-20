"""Greedy streaming completion over an exclusively owned chain pipeline.

Mirrors the two-rank HostCompletion contract (token events, finish_reason,
usage, single-owner claim, cancellation between commands) on top of the
three-rank ChainPipeline: HTTP threads stream; only the owning thread drives
the coordinator. Generation convention is the qualified executor one:
prefill's argmax is the first token, each decode consumes the previous output.
On any execution failure the epoch poisons on every rank, so close() is the
only recovery (fresh workers and pipeline required).
"""
from contextlib import closing
import threading

from .chain_pipeline import ChainPipeline
from .hybrid_pipeline import PipelineCancelled


class ChainCompletion:
    """Own a ready empty pipeline; successful requests release their slots."""

    def __init__(self, pipeline, *, eos_ids=()):
        if not isinstance(pipeline, ChainPipeline):
            raise ValueError('chain completion requires a ChainPipeline')
        if pipeline.failed or pipeline.closed:
            raise ValueError('chain completion requires a ready pipeline')
        if any(pipeline.state.lengths):
            raise ValueError('chain completion requires empty cursors')
        if (not isinstance(eos_ids, (list, tuple, set, frozenset)) or len(eos_ids) > 256
                or any(type(t) is not int or not 0 <= t < pipeline.vocab_size
                       for t in eos_ids)):
            raise ValueError('invalid chain completion EOS policy')
        if getattr(pipeline, '_completion_owner', None) is not None:
            raise ValueError('chain completion requires an unclaimed pipeline')
        pipeline._completion_owner = object()  # type: ignore[attr-defined]
        self.pipeline = pipeline
        self.eos_ids = frozenset(eos_ids)
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def close(self):
        self.stopped.set()
        if self.pipeline.failed:
            # A poisoned epoch cannot accept STOP (every rank rejects begin);
            # workers exit on their own once they see the poisoned state.
            self.pipeline.closed = True
        else:
            self.pipeline.close()

    cleanup = close

    def _check_cancelled(self, cancelled):
        if self.stopped.is_set() or (cancelled is not None and cancelled()):
            raise PipelineCancelled('chain completion cancelled')

    def generate_batch(self, rows, *, max_new_tokens=128, cancelled=None):
        """Stream greedy rows: {row, token_id} per step, then usage per row.

        rows[i] prefills slot i. Row length plus output limit must fit the
        session context; finished slots release for reuse within the stream.
        Validation errors (raised before any command) leave the session
        healthy; errors once streaming starts discard it via close().
        """
        if cancelled is not None and not callable(cancelled):
            raise ValueError('cancellation must be callable')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('chain completion is active')
        try:
            rows, limits = self._prepare(rows, max_new_tokens)
        except BaseException:
            self.lock.release()
            raise
        primary = None
        try:
            sampled = [None] * len(rows)
            for slot, tokens in enumerate(rows):
                self._check_cancelled(cancelled)
                # The head's PREFILL argmax predicts the token after the
                # prompt (qualified against the single-rank executor).
                sampled[slot] = self.pipeline.prefill(tokens, slot=slot)[0]
            active = list(range(len(rows)))
            counts = [0] * len(rows)
            reasons = ['length'] * len(rows)
            while active:
                self._check_cancelled(cancelled)
                survivors = []
                next_tokens = []
                finished = []
                for row in active:
                    self._check_cancelled(cancelled)
                    token = sampled[row]
                    counts[row] += 1
                    yield dict(row=row, token_id=token)
                    if token in self.eos_ids:
                        reasons[row] = 'stop'
                        finished.append(row)
                    elif counts[row] >= limits[row]:
                        finished.append(row)
                    else:
                        survivors.append(row)
                        next_tokens.append(token)
                self._check_cancelled(cancelled)
                if finished:
                    self.pipeline.release(finished)
                active = survivors
                if active:
                    self._check_cancelled(cancelled)
                    outputs = self.pipeline.decode_batch(
                        next_tokens, [row for row in active])
                    for row, token in zip(active, outputs):
                        sampled[row] = token
            for row, tokens in enumerate(rows):
                yield dict(row=row, finish_reason=reasons[row], usage=dict(
                    prompt_tokens=len(tokens), completion_tokens=counts[row],
                    total_tokens=len(tokens) + counts[row]))
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                if primary is not None:
                    try:
                        self.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            'chain completion cleanup failed; retain owner and '
                            f'retry close: {type(cleanup_error).__name__}')
            finally:
                self.lock.release()

    def generate(self, tokens, *, max_new_tokens=128, cancelled=None):
        """Stream one greedy row without the row field."""
        if type(max_new_tokens) is not int:
            raise ValueError('single chain completion requires an integer output limit')
        with closing(self.generate_batch([tokens], max_new_tokens=max_new_tokens,
                                         cancelled=cancelled)) as stream:
            for event in stream:
                yield {key: value for key, value in event.items() if key != 'row'}

    def _prepare(self, rows, max_new_tokens):
        """Validate rows/limits against the session geometry."""
        pipeline = self.pipeline
        if (not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= pipeline.state.slots
                or any(not isinstance(row, (list, tuple)) or not row
                       or any(type(t) is not int or not 0 <= t < pipeline.vocab_size
                              for t in row)
                       for row in rows)):
            raise ValueError('invalid chain completion prompt rows')
        limits = ([max_new_tokens] * len(rows) if type(max_new_tokens) is int
                  else list(max_new_tokens))
        if (not isinstance(limits, list) or len(limits) != len(rows)
                or any(type(limit) is not int or not 1 <= limit <= 4096
                       for limit in limits)):
            raise ValueError('invalid chain completion output limits')
        for row, limit in zip(rows, limits):
            if len(row) + limit > pipeline.context:
                raise ValueError('chain completion prompt plus output exceeds context')
        return [list(row) for row in rows], limits
