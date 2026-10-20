"""Experimental three-rank chain pipeline coordinator (#536 gate 4, cycle 2).

Wires the cycle-1 control channel (distributed_chain) to the stage protocol
(stage_protocol) and real hybrid shards: the coordinator (rank 0) issues
encoded commands, executes its own shard inline, and every rank's StageState
commits the request cursor only after the MIN acknowledgement across ALL
stages succeeds. Any execution failure poisons the epoch on every rank
before any cursor advances — the hybrid_stage.py ownership contract.

Payload contract (matches the two-rank HostHybridWorker shape):
- hidden states travel as contiguous FP32 [rows, hidden_width] tensors;
- the head rank returns `rows` int64 argmax token ids per round, for both
  PREFILL (next-token after the prompt) and DECODE. Full-vocab logit
  transport for head-side sampling is a later cycle.
- Supported operations: PREFILL, DECODE, STOP. RELEASE/PREFILL_BATCH/
  PREFILL_CHUNK/SWITCH remain with the two-rank machinery for now.

Control plane only: model loading, identity agreement and HTTP serving
stay with the caller. CPU tests drive the same calls through the
FakeBus/FakeTransport pair in tests/test_distributed_chain.py.
"""
from __future__ import annotations

import threading

import torch

from .distributed_chain import ChainChannel
from .stage_protocol import (DECODE, FAILED_TOKEN, PREFILL, STOP, StageState,
                             Command, token_rows)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ChainStageWorker:
    """Worker-side execution of chain rounds (any rank; rank 0 optional).

    The coordinator drives rank 0 through ChainPipeline's own round method
    instead of this loop — but the class stays rank-agnostic so tests can
    run any geometry. Ranks > 0 run `run_forever` on their own thread.
    """

    def __init__(self, stage, channel):
        if not isinstance(channel, ChainChannel):
            raise ValueError('chain worker needs a ChainChannel')
        state = channel.state
        if (not isinstance(state, StageState) or stage.closed
                or stage.context != state.context or stage.max_batch != state.slots
                or stage.c['hidden_size'] <= 0):
            raise ValueError('chain worker needs a fresh stage matching the channel state')
        head = channel.role.owns_head
        if head and not stage.last:
            raise ValueError('the head rank owns the output stage')
        if not head and stage.last:
            raise ValueError('a non-head rank cannot own the output stage')
        if channel.role.is_coordinator and not stage.first:
            raise ValueError('the coordinator owns the embedding stage')
        self.stage = stage
        self.channel = channel
        self.rank = channel.role.rank
        self.width = stage.c['hidden_size']
        self.vocab_size = stage.c['vocab_size']
        self.head = head
        self.failed = False
        self.closed = False
        self.stopped = threading.Event()

    @torch.inference_mode()
    def _stage_forward(self, command, tokens=None):
        """Execute this shard; returns the hidden state [rows, width] FP32."""
        stage = self.stage
        rows = command.rows
        if command.operation == DECODE:
            slots = torch.tensor([entry[0] for entry in command.entries], dtype=torch.int64)
            positions = torch.tensor([[entry[2]] for entry in command.entries],
                                     dtype=torch.int64)
            prefill = False
        elif command.operation == PREFILL:
            slot, _, _ = command.entries[0]
            slots = torch.tensor([slot], dtype=torch.int64)
            positions = torch.arange(rows)[None]
            prefill = True
        else:
            raise ValueError('unsupported chain stage operation')
        if self.rank == 0:
            values = torch.tensor(tokens, dtype=torch.int64)
            if command.operation == DECODE:
                hidden = stage.embedding(values[:, None])
            else:
                hidden = stage.embedding(values[None])
        else:
            received = torch.zeros((rows, self.width), dtype=torch.float32)
            self.channel.receive_hidden(received)
            if not torch.isfinite(received).all():
                raise ValueError('nonfinite chain received activation')
            # DECODE rows are batch slots: [rows, 1, width]. PREFILL rows are
            # one packed sequence: [1, rows, width].
            hidden = received[:, None, :] if command.operation == DECODE else received[None]
        device = self.stage.actual_device
        with torch.cuda.device(device) if device.type == 'cuda' else _nullcontext():
            result = stage.forward_layers(hidden, positions, slots, prefill)
        hidden_out = result.reshape(-1, self.width)[-rows:].to(torch.float32).contiguous()
        if hidden_out.shape != (rows, self.width) or not torch.isfinite(hidden_out).all():
            raise ValueError('invalid chain stage activation')
        return hidden_out

    def run_round(self, tokens=None):
        """Execute the channel's pending command; the head returns argmax tokens.

        Any execution failure funnels through finish(success=False): the
        worker still emits a fallback payload so the chain stays alive, and
        the MIN ack poisons every rank before any cursor advances.
        """
        command = self.channel.state.pending
        success = True
        if command.operation == STOP:
            # Control-only record: no execution, no payload, no tokens.
            self.channel.finish(command)
            self.closed = True
            self.stopped.set()
            return
        try:
            hidden = self._stage_forward(command, tokens)
            if self.stage.last:
                rows = token_rows(command)
                logits = self.stage.output_logits(hidden[-rows:])
                if logits.shape != (rows, self.vocab_size):
                    raise ValueError('invalid chain output logits')
                self.channel.return_tokens(logits.argmax(-1).to(torch.int64))
            if not self.stage.last:
                self.channel.forward_hidden(hidden)
        except BaseException:
            success = False
            self.failed = True
            # Keep the chain alive so every rank reaches the MIN ack.
            try:
                if self.stage.last:
                    self.channel.return_tokens(
                        torch.full((token_rows(command),), FAILED_TOKEN, dtype=torch.int64))
                elif not self.stage.first:
                    self.channel.forward_hidden(
                        torch.zeros((command.rows, self.width), dtype=torch.float32))
            except RuntimeError:
                pass
        try:
            self.channel.finish(command, success=success)
        except BaseException:
            self.failed = True
            raise
        # Mirror the two-rank worker: keep the stage's cursor view in lockstep.
        self.stage.lengths[:] = self.channel.state.lengths
        self.stage.length = self.channel.state.lengths[0]

    def run_forever(self):
        """Worker-rank loop: rounds until STOP closes the epoch or it poisons."""
        while not self.closed and not self.stopped.is_set():
            try:
                self.channel.begin(None)
            except (ValueError, RuntimeError):
                if self.channel.state.closed or self.channel.state.failed:
                    break
                raise
            try:
                self.run_round()
            except BaseException:
                # A poisoned epoch needs fresh workers; exit the loop quietly.
                self.failed = True
                self.closed = True
                self.stopped.set()
                return
            if self.channel.state.closed:
                self.closed = True
                self.stopped.set()

    def close(self):
        self.stopped.set()


class ChainPipeline:
    """Coordinator-side handle (rank 0): issues commands, owns request cursors.

    The coordinator's StageState IS the request cursor authority; every
    worker stage receives identical commands and commits in lockstep, so
    the cursors only advance when the whole chain acknowledged a round.
    """

    def __init__(self, stage, channel):
        if not isinstance(channel, ChainChannel):
            raise ValueError('chain pipeline needs a ChainChannel')
        if not channel.role.is_coordinator:
            raise ValueError('only rank 0 drives the chain pipeline')
        state = channel.state
        if (not isinstance(state, StageState) or stage.closed
                or stage.context != state.context or stage.max_batch != state.slots):
            raise ValueError('chain pipeline needs a fresh stage matching the channel state')
        if not stage.first or stage.last:
            raise ValueError('the coordinator owns the embedding stage, not the head')
        self.stage = stage
        self.channel = channel
        self.state = state
        self.width = stage.c['hidden_size']
        self._stage0_worker = ChainStageWorker(stage, channel)
        self.failed = False
        self.closed = False

    def _command(self, operation, rows, entries):
        return Command(self.state.epoch, self.state.sequence, operation, rows, entries)

    def _run_stage0(self, command, tokens):
        """Execute stage 0; on failure send a fallback payload and report it.

        The fallback keeps downstream ranks alive so the whole chain reaches
        the MIN ack and poisons together instead of deadlocking.
        """
        try:
            return self._stage0_worker._stage_forward(command, tokens), None
        except BaseException as error:
            try:
                self.channel.forward_hidden(
                    torch.zeros((command.rows, self.width), dtype=torch.float32))
            except RuntimeError:
                pass
            return None, error

    def prefill(self, tokens, slot=0):
        """Prompt one slot; returns the head's argmax next-token id."""
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        if (not tokens or type(slot) is not int or not 0 <= slot < self.state.slots
                or any(type(t) is not int for t in tokens)
                or len(tokens) > self.state.context):
            raise ValueError('invalid chain prefill prompt or slot')
        generation = self.state.generations[slot] + 1
        command = self._command(PREFILL, len(tokens), ((slot, generation, 0),))
        return self._round(command, tokens)

    def decode_batch(self, tokens, slots):
        """One decode step for a batch of slots; returns argmax token ids."""
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        if (not tokens or len(tokens) != len(slots)
                or any(type(t) is not int or type(s) is not int for t, s in zip(tokens, slots))
                or len(set(slots)) != len(slots)):
            raise ValueError('invalid chain decode batch')
        entries = []
        for token, slot in zip(tokens, slots):
            if not 0 <= slot < self.state.slots or not 0 < self.state.lengths[slot] < self.state.context:
                raise ValueError('chain decode slot has no live cursor')
            entries.append((slot, self.state.generations[slot], self.state.lengths[slot]))
        command = self._command(DECODE, len(slots), tuple(entries))
        return self._round(command, tokens)

    def _round(self, command, tokens):
        success = True
        result = None
        failure = None
        received = None
        began = False
        try:
            accepted = self.channel.begin(command.encode())
            began = True
            if accepted != command:
                raise RuntimeError('chain stage accepted a different command')
            hidden, error = self._run_stage0(command, tokens)
            if error is not None:
                raise RuntimeError('chain coordinator stage failed') from error
            self.channel.forward_hidden(hidden)
            received = torch.full((token_rows(command),), FAILED_TOKEN, dtype=torch.int64)
            self.channel.receive_tokens(received)
            if any(token == FAILED_TOKEN for token in received.tolist()):
                raise RuntimeError('chain head reported stage failure')
            result = received.tolist()
        except BaseException as error:
            success = False
            failure = error
            # Drain the token round the head still owes so every rank reaches
            # the MIN ack; the round's result is discarded either way. If the
            # tokens already arrived (head-side failure), or the round never
            # began, there is nothing to drain.
            if began and received is None:
                try:
                    self.channel.receive_tokens(
                        torch.full((token_rows(command),), FAILED_TOKEN, dtype=torch.int64))
                except RuntimeError:
                    pass
        try:
            self.channel.finish(command, success=success)
        except RuntimeError:
            self.failed = True
            raise
        if not success:
            self.failed = True
            assert failure is not None
            raise failure
        return result

    def close(self):
        """Issue STOP; workers close their stages and exit their loops."""
        if self.closed:
            return
        try:
            command = self._command(STOP, 0, ())
            self.channel.begin(command.encode())
            self.channel.finish(command)
        finally:
            self.failed = True
            self.closed = True
