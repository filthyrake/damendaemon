"""Bounded control records for experimental two-worker execution.

Records travel as fixed int64 tensors, never Python objects. Model identity is
established separately before opening an epoch. Only one operation may be in
flight; cache state commits after both stages acknowledge completion. Any
execution or transport failure poisons the epoch and requires fresh workers.
"""
from dataclasses import dataclass

MAGIC = 0x444C4C4D
VERSION = 1
MAX_SLOTS = 4
HEADER_WORDS = 7 + 3 * MAX_SLOTS
PREFILL, DECODE, RELEASE, STOP = range(1, 5)
MAX_INTEGER = (1 << 63) - 1


@dataclass(frozen=True)
class Command:
    epoch: int
    sequence: int
    operation: int
    rows: int
    # Each entry is (slot, generation, position before execution).
    entries: tuple

    def encode(self):
        if len(self.entries) > MAX_SLOTS:
            raise ValueError('too many stage slots')
        words = [MAGIC, VERSION, self.epoch, self.sequence, self.operation,
                 self.rows, len(self.entries)]
        for entry in self.entries:
            if len(entry) != 3:
                raise ValueError('invalid stage slot record')
            words.extend(entry)
        words.extend([-1] * (HEADER_WORDS - len(words)))
        # Validate locally too, before constructing a GPU tensor.
        decode(words)
        return words


def decode(words):
    if (len(words) != HEADER_WORDS or
            any(type(v) is not int or not -1 <= v <= MAX_INTEGER for v in words)):
        raise ValueError('invalid stage control record')
    magic, version, epoch, sequence, operation, rows, count = words[:7]
    if magic != MAGIC or version != VERSION:
        raise ValueError('unsupported stage control protocol')
    if epoch < 1 or sequence < 1 or operation not in (PREFILL, DECODE, RELEASE, STOP):
        raise ValueError('invalid stage epoch, sequence or operation')
    if not 0 <= count <= MAX_SLOTS or rows < 0:
        raise ValueError('invalid stage shape')
    if any(v != -1 for v in words[7 + 3 * count:]):
        raise ValueError('noncanonical stage control padding')
    entries = tuple(tuple(words[i:i + 3]) for i in range(7, 7 + 3 * count, 3))
    if (len({entry[0] for entry in entries}) != count or
            any(not 0 <= slot < MAX_SLOTS or generation < 1 or position < 0
                for slot, generation, position in entries)):
        raise ValueError('invalid stage slot ownership')
    return Command(epoch, sequence, operation, rows, entries)


class StageState:
    def __init__(self, epoch, context, slots=MAX_SLOTS):
        if (type(epoch) is not int or not 1 <= epoch <= MAX_INTEGER or
                type(context) is not int or not 1 <= context <= 512 or
                type(slots) is not int or not 1 <= slots <= MAX_SLOTS):
            raise ValueError('unsupported stage session bounds')
        self.epoch, self.context, self.slots = epoch, context, slots
        self.sequence = 1
        self.generations = [0] * slots
        self.lengths = [0] * slots
        self.pending = None
        self.failed = False
        self.closed = False

    def begin(self, words):
        if self.failed or self.closed or self.pending is not None:
            raise RuntimeError('stage epoch is not ready')
        command = decode(words)
        if command.epoch != self.epoch or command.sequence != self.sequence:
            raise ValueError('stale stage epoch or sequence')
        if any(slot >= self.slots for slot, _, _ in command.entries):
            raise ValueError('slot exceeds stage capacity')
        if command.operation == PREFILL:
            if len(command.entries) != 1 or not 1 <= command.rows <= self.context:
                raise ValueError('invalid stage prefill shape')
            slot, generation, position = command.entries[0]
            if generation != self.generations[slot] + 1 or position != 0:
                raise ValueError('stale stage prefill generation')
        elif command.operation in (DECODE, RELEASE):
            if not command.entries or command.rows != (len(command.entries) if command.operation == DECODE else 0):
                raise ValueError('invalid stage decode or release shape')
            for slot, generation, position in command.entries:
                if (generation != self.generations[slot] or
                        position != self.lengths[slot] or position == 0 or
                        (command.operation == DECODE and position >= self.context)):
                    raise ValueError('stale or exhausted stage slot')
        elif command.rows != 0 or command.entries:
            raise ValueError('invalid stage stop record')
        self.pending = command
        return command

    def commit(self, command):
        if self.failed or self.pending is None or command != self.pending:
            raise RuntimeError('stage completion does not own pending work')
        for slot, generation, _ in command.entries:
            if command.operation == PREFILL:
                self.generations[slot] = generation
                self.lengths[slot] = command.rows
            elif command.operation == DECODE:
                self.lengths[slot] += 1
            elif command.operation == RELEASE:
                self.lengths[slot] = 0
        self.closed = command.operation == STOP
        self.sequence += 1
        self.pending = None

    def poison(self):
        self.failed = True
        self.pending = None
        self.lengths = [0] * self.slots
