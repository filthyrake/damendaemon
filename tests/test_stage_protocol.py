import unittest
from dllm.stage_protocol import Command, StageState, decode, PREFILL, DECODE, RELEASE, STOP


class StageProtocolTests(unittest.TestCase):
    def run_command(self, state, operation, rows, entries):
        command = Command(state.epoch, state.sequence, operation, rows, tuple(entries))
        accepted = state.begin(command.encode())
        state.commit(accepted)
        return command

    def test_reordered_shrinking_batch_release_and_reuse(self):
        state = StageState(7, 512)
        self.run_command(state, PREFILL, 31, [(2, 1, 0)])
        self.run_command(state, PREFILL, 255, [(0, 1, 0)])
        self.run_command(state, DECODE, 2, [(2, 1, 31), (0, 1, 255)])
        self.run_command(state, RELEASE, 0, [(2, 1, 32)])
        self.run_command(state, DECODE, 1, [(0, 1, 256)])
        self.run_command(state, PREFILL, 11, [(2, 2, 0)])
        self.assertEqual(state.lengths, [257, 0, 11, 0])
        with self.assertRaises(ValueError):
            state.begin(Command(7, state.sequence, DECODE, 1, ((2, 1, 11),)).encode())
        self.assertEqual(state.lengths, [257, 0, 11, 0])

    def test_validation_precedes_state_mutation(self):
        state = StageState(1, 8)
        self.run_command(state, PREFILL, 8, [(0, 1, 0)])
        bad = [Command(2, 2, DECODE, 1, ((0, 1, 8),)),
               Command(1, 1, RELEASE, 0, ((0, 1, 8),)),
               Command(1, 2, DECODE, 1, ((0, 1, 8),)),
               Command(1, 2, PREFILL, 9, ((0, 2, 0),)),
               Command(1, 2, RELEASE, 0, ((0, 1, 7),)),
               Command(1, 2, DECODE, 2, ((0, 1, 8), (0, 1, 8)))]
        for command in bad:
            with self.subTest(command=command), self.assertRaises(ValueError):
                state.begin(command.encode())
            self.assertIsNone(state.pending)
            self.assertEqual(state.sequence, 2)
            self.assertEqual(state.lengths, [8, 0, 0, 0])

    def test_commit_after_ack_and_poison_after_failure(self):
        state = StageState(1, 8)
        command = state.begin(Command(1, 1, PREFILL, 3, ((0, 1, 0),)).encode())
        self.assertEqual(state.lengths, [0] * 4)
        with self.assertRaises(RuntimeError):
            state.begin(command.encode())
        state.poison()
        with self.assertRaises(RuntimeError):state.commit(command)
        with self.assertRaises(RuntimeError):state.begin(command.encode())
        replacement = StageState(2, 8)
        with self.assertRaises(ValueError):replacement.begin(command.encode())
        self.run_command(replacement, PREFILL, 3, [(0, 1, 0)])

    def test_canonical_fixed_size_wire_and_stop(self):
        command = Command(1, 1, STOP, 0, ())
        self.assertEqual(decode(command.encode()), command)
        for index, value in [(0, 0), (1, 99), (6, 5), (7, 0), (2, True), (3, 1 << 63)]:
            words = command.encode(); words[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):decode(words)
        with self.assertRaises(ValueError):decode(command.encode()[:-1])
        state = StageState(1, 8)
        self.run_command(state, STOP, 0, [])
        self.assertTrue(state.closed)
        with self.assertRaises(RuntimeError):state.begin(command.encode())
