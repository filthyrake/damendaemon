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
from .stage_protocol import (DECODE, FAILED_TOKEN, PING, PREFILL, PREFILL_CHUNK,
                             RELEASE, STOP, SWITCH, StageState, Command,
                             token_rows)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def run_chain_round(channel, worker, tokens=None):
    """One round of the broadcast chain; EVERY rank runs the identical
    collective sequence — hidden broadcasts src=0..head in rank order, the
    token broadcast src=head, then the MIN ack.

    Cross-node P2P send/recv deadlocks the three-rank ring (gate-4 cycle 1),
    so all transfers are broadcasts. A failing rank emits a fallback payload
    (zeros hidden / FAILED_TOKEN tokens) instead of skipping its exchange,
    then finish()'s MIN ack poisons every rank before any cursor advances.

    Returns the head's token ids on the coordinator, None elsewhere.
    Raises on the coordinator when the round failed or the epoch poisoned.
    """
    command = channel.state.pending
    head_rank = channel.role.head_rank
    received = None
    failure = None
    head_tokens = None
    for source in range(head_rank + 1):
        is_tokens = source == head_rank
        payload = None
        if source == worker.rank:
            try:
                payload = worker._stage_forward(command, tokens, received)
            except BaseException as error:
                failure = error
                worker.failed = True
                payload = worker._fallback(command, is_tokens)
        if is_tokens:
            if payload is None:
                payload = torch.empty((token_rows(command),), dtype=torch.int64,
                                      device=worker.device)
            channel.exchange_tokens(payload, source)
            if channel.role.is_coordinator:
                head_tokens = payload
        else:
            if payload is None:
                payload = torch.zeros((command.rows, worker.width),
                                      dtype=torch.float32, device=worker.device)
            channel.exchange_hidden(payload, source)
            received = payload
    try:
        channel.finish(command, success=failure is None)
    except RuntimeError:
        worker.failed = True
        raise
    if failure is not None:
        worker.failed = True
        raise failure
    if channel.role.is_coordinator:
        if any(token == FAILED_TOKEN for token in head_tokens.tolist()):
            raise RuntimeError('chain head reported stage failure')
        return head_tokens.tolist()
    return None


class ChainStageWorker:
    """Worker-side execution of chain rounds (any rank; rank 0 optional).

    The coordinator drives rank 0 through ChainPipeline's own round method
    instead of this loop — but the class stays rank-agnostic so tests can
    run any geometry. Ranks > 0 run `run_forever` on their own thread.
    """

    def __init__(self, stage, channel, *, device=None, shards=None):
        if not isinstance(channel, ChainChannel):
            raise ValueError('chain worker needs a ChainChannel')
        state = channel.state
        if shards is not None:
            if (not isinstance(shards, list) or len(shards) != state.models
                    or any(not isinstance(shard, dict)
                            or any(type(name) is not str for name in shard)
                            for shard in shards)):
                raise ValueError('chain worker shards must match the model table')
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
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.shards = shards
        self.failed = False
        self.closed = False
        self.stopped = threading.Event()

    @torch.inference_mode()
    def _stage_forward(self, command, tokens=None, received=None):
        """Execute this shard; returns this rank's round payload.

        Ranks before the head return the hidden state [rows, width] FP32;
        the head rank returns `token_rows(command)` int64 argmax ids.
        """
        stage = self.stage
        rows = command.rows
        if command.operation == DECODE:
            slots = torch.tensor([entry[0] for entry in command.entries],
                                 dtype=torch.int64, device=self.device)
            positions = torch.tensor([[entry[2]] for entry in command.entries],
                                     dtype=torch.int64, device=self.device)
            prefill = False
            prefix = 0
        elif command.operation in (PREFILL, PREFILL_CHUNK):
            slot, _, start = command.entries[0]
            slots = torch.tensor([slot], dtype=torch.int64, device=self.device)
            positions = torch.arange(start, start + rows,
                                     device=self.device)[None]
            prefill = True
            prefix = start
        else:
            raise ValueError('unsupported chain stage operation')
        if self.rank == 0:
            values = torch.tensor(tokens, dtype=torch.int64, device=self.device)
            if command.operation == DECODE:
                hidden = stage.embedding(values[:, None])
            else:
                hidden = stage.embedding(values[None])
        else:
            if received is None or received.shape != (rows, self.width):
                raise ValueError('chain stage missing its broadcast hidden input')
            if not torch.isfinite(received).all():
                raise ValueError('nonfinite chain received activation')
            # DECODE rows are batch slots: [rows, 1, width]. PREFILL rows are
            # one packed sequence: [1, rows, width].
            hidden = received[:, None, :] if command.operation == DECODE else received[None]
        device = self.stage.actual_device
        with torch.cuda.device(device) if device.type == 'cuda' else _nullcontext():
            result = stage.forward_layers(hidden, positions, slots, prefill,
                                          prefix)
        if self.stage.last:
            count = token_rows(command)
            logits = stage.output_logits(result.reshape(-1, self.width)[-count:])
            if logits.shape != (count, self.vocab_size):
                raise ValueError('invalid chain output logits')
            return logits.argmax(-1).to(torch.int64)
        hidden_out = result.reshape(-1, self.width)[-rows:].to(torch.float32).contiguous()
        if hidden_out.shape != (rows, self.width) or not torch.isfinite(hidden_out).all():
            raise ValueError('invalid chain stage activation')
        return hidden_out

    def _fallback(self, command, is_tokens):
        """Deadlock-free failure payload so every rank reaches the MIN ack."""
        if is_tokens:
            return torch.full((token_rows(command),), FAILED_TOKEN,
                              dtype=torch.int64, device=self.device)
        return torch.zeros((command.rows, self.width), dtype=torch.float32,
                           device=self.device)

    def _clear_slots(self, command):
        """Zero a released slot's KV/conv/recurrent tensors (two-rank parity)."""
        for slot, _, _ in command.entries:
            for tensor in self.stage.kv.values():
                members = tensor if isinstance(tensor, tuple) else (tensor,)
                for member in members:
                    member[:, slot].zero_()
            for group in (self.stage.conv, self.stage.state):
                for tensor in group.values():
                    tensor[slot].zero_()

    def _apply_shard(self, index):
        """Copy a pre-agreed model's shard weights into the resident stage.

        Unquantized only: packed Q8/Q4 scales are bound to weight tensor ids
        inside the quantized Linear (the two-rank stage_rebind machinery
        owns that path). Same geometry is validated BEFORE any copy so a
        rejection cannot leave a half-overwritten stage.
        """
        if self.shards is None:
            raise ValueError('this chain has no alternate model table')
        if type(index) is not int or not 0 <= index < len(self.shards):
            raise ValueError('invalid chain shard index')
        shard = self.shards[index]
        if set(shard) != set(self.stage.w):
            raise ValueError('chain shard does not cover the stage weights')
        for name, target in self.stage.w.items():
            source = shard[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f'chain shard {name} geometry mismatch')
        for name, target in self.stage.w.items():
            target.copy_(shard[name])

    def run_round(self, tokens=None):
        """Execute the channel's pending command via the broadcast chain."""
        command = self.channel.state.pending
        if command.operation in (STOP, PING):
            # Control-only records: no execution, no payload, no tokens.
            # STOP closes the loop; PING (idle heartbeat) keeps it alive.
            self.channel.finish(command)
            if command.operation == STOP:
                self.closed = True
                self.stopped.set()
            return
        if command.operation == RELEASE:
            # Control-only round: clear the released slots' KV/conv/recurrent
            # state so the next prefill starts from a clean slot.
            self._clear_slots(command)
            self.channel.finish(command)
            self.stage.lengths[:] = self.channel.state.lengths
            self.stage.length = self.channel.state.lengths[0]
            return
        if command.operation == SWITCH:
            # Resident model swap: copy the pre-agreed shard in place (all
            # slots must be empty — StageState.begin validated that), then
            # the MIN ack commits model_index/epoch/generations everywhere
            # or poisons the epoch on any failure.
            success = True
            try:
                self._apply_shard(command.rows)
            except BaseException:
                success = False
                self.failed = True
            self.channel.finish(command, success=success)
            self.stage.lengths[:] = self.channel.state.lengths
            self.stage.length = self.channel.state.lengths[0]
            if not success:
                raise RuntimeError('chain stage model switch failed')
            return
        try:
            run_chain_round(self.channel, self, tokens)
        except BaseException:
            # A poisoned epoch needs fresh workers; exit the loop quietly.
            self.failed = True
            self.closed = True
            self.stopped.set()
            return
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

    def __init__(self, stage, channel, *, device=None, shards=None):
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
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self._stage0_worker = ChainStageWorker(stage, channel, device=self.device,
                                              shards=shards)
        self.failed = False
        self.closed = False

    def _command(self, operation, rows, entries):
        return Command(self.state.epoch, self.state.sequence, operation, rows, entries)

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

    def prefill_chunk(self, tokens, slot, start):
        """Continue one slot's prompt at `start`; returns the head's argmax.

        Mirrors HybridExecutor.prefill_chunk ownership: start==0 claims a
        fresh generation; start>0 must equal the slot's committed cursor.
        The returned token is a protocol sample — only the FINAL chunk's
        token is an output candidate (same convention as the scheduler).
        """
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        if (not tokens or type(slot) is not int or not 0 <= slot < self.state.slots
                or type(start) is not int or start < 0
                or any(type(t) is not int for t in tokens)
                or start + len(tokens) > self.state.context):
            raise ValueError('invalid chain prefill chunk')
        if start == 0:
            if self.state.lengths[slot]:
                raise ValueError('chain prefill chunk requires an empty slot')
            generation = self.state.generations[slot] + 1
        else:
            if self.state.lengths[slot] != start:
                raise ValueError('chain prefill chunk must continue the cursor')
            generation = self.state.generations[slot]
        command = self._command(PREFILL_CHUNK, len(tokens),
                                ((slot, generation, start),))
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
        try:
            accepted = self.channel.begin(command.encode())
            if accepted != command:
                raise RuntimeError('chain stage accepted a different command')
            result = run_chain_round(self.channel, self._stage0_worker, tokens)
        except BaseException as error:
            self.failed = True
            raise error
        return result

    @property
    def vocab_size(self):
        return self._stage0_worker.vocab_size

    @property
    def context(self):
        return self.state.context

    def ping(self):
        """Control-only heartbeat round; keeps idle workers in lockstep."""
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        command = self._command(PING, 0, ())
        try:
            self.channel.begin(command.encode())
            self.channel.finish(command)
        except BaseException:
            self.failed = True
            raise
        return True

    def switch(self, index):
        """Swap the resident model to a pre-agreed table index.

        Requires every slot released; the SWITCH round copies each rank's
        shard in place and the MIN ack commits model_index + a fresh epoch
        (generations reset) everywhere, or poisons on any failure. Returns
        False when already current (two-rank switch_model semantics).
        """
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        if type(index) is not int or not 0 <= index < self.state.models:
            raise ValueError('invalid chain model index')
        if index == self.state.model_index:
            return False
        if any(self.state.lengths):
            raise ValueError('model switch requires released request slots')
        command = self._command(SWITCH, index, ())
        success = True
        try:
            self.channel.begin(command.encode())
            self._stage0_worker._apply_shard(index)
        except BaseException:
            success = False
        try:
            self.channel.finish(command, success=success)
        except BaseException:
            self.failed = True
            raise
        self._stage0_worker.stage.lengths[:] = self.state.lengths
        self._stage0_worker.stage.length = self.state.lengths[0]
        if not success:
            self.failed = True
            raise RuntimeError('chain model switch failed')
        return True

    def release(self, slots):
        """Control-only round: free finished slots for the next prefill.

        Validates on the coordinator (against the shared StageState content)
        so a rejected release cannot strand the workers outside the ack.
        """
        if self.failed or self.closed:
            raise RuntimeError('chain pipeline is not ready')
        if (not isinstance(slots, (list, tuple)) or not slots
                or len(set(slots)) != len(slots)
                or any(type(s) is not int or not 0 <= s < self.state.slots
                       for s in slots)
                or any(not self.state.lengths[s] for s in slots)):
            raise ValueError('invalid chain release slot selection')
        command = self._command(RELEASE, 0,
                                tuple((slot, self.state.generations[slot],
                                       self.state.lengths[slot]) for slot in slots))
        try:
            self.channel.begin(command.encode())
            self.channel.finish(command)
        except BaseException:
            self.failed = True
            raise
        self._stage0_worker.stage.lengths[:] = self.state.lengths
        self._stage0_worker.stage.length = self.state.lengths[0]

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
