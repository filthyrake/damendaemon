import unittest
from damendaemon.stage_protocol import Command, StageState, decode, PREFILL, DECODE, RELEASE, STOP, PING, PREFILL_BATCH, PREFILL_CHUNK, SWITCH, KV_HANDOFF, token_rows


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

    def test_single_prefill_cannot_replace_live_receiver_ownership(self):
        state=StageState(7,32,2)
        self.run_command(state,PREFILL,4,[(1,1,0)])
        replacement=Command(7,2,PREFILL,3,((1,2,0),))
        with self.assertRaises(ValueError):state.begin(replacement.encode())
        self.assertEqual(state.lengths,[0,4])
        self.assertEqual(state.generations,[0,1])
        self.assertEqual(state.sequence,2)
        self.assertIsNone(state.pending)
        self.run_command(state,RELEASE,0,[(1,1,4)])
        self.run_command(state,PREFILL,3,[(1,2,0)])
        self.assertEqual(state.lengths,[0,3])
        self.assertEqual(state.generations,[0,2])

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
        for index, value in [(0, 0), (1, 99), (1, 1), (1, 2), (1, 3), (1, 4), (6, 5), (7, 0), (2, True), (3, 1 << 63)]:
            words = command.encode(); words[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):decode(words)
        with self.assertRaises(ValueError):decode(command.encode()[:-1])
        state = StageState(1, 8)
        self.run_command(state, STOP, 0, [])
        self.assertTrue(state.closed)
        with self.assertRaises(RuntimeError):state.begin(command.encode())


class HeartbeatTests(unittest.TestCase):
    def test_heartbeat_preserves_live_slot_and_advances_sequence_after_ack(self):
        state=StageState(1,512,1)
        prefill=state.begin(Command(1,1,PREFILL,8,((0,1,0),)).encode());state.commit(prefill)
        ping=state.begin(Command(1,2,PING,0,()).encode())
        self.assertEqual(state.sequence,2)
        self.assertEqual(state.lengths,[8]);self.assertEqual(state.generations,[1])
        state.commit(ping)
        self.assertEqual(state.sequence,3);self.assertFalse(state.closed)
        self.assertEqual(state.lengths,[8]);self.assertEqual(state.generations,[1])
        with self.assertRaises(ValueError):state.begin(ping.encode())
        for rows,entries in ((1,()),(0,((0,1,8),))):
            with self.assertRaises(ValueError):state.begin(Command(1,3,PING,rows,entries).encode())
        self.assertIsNone(state.pending);self.assertEqual(state.lengths,[8])


class PackedPrefillProtocolTests(unittest.TestCase):
    def test_lengths_and_generations_commit_only_after_acknowledgement(self):
        state=StageState(1,32,4)
        command=Command(1,1,PREFILL_BATCH,11,((3,1,4),(0,1,7)))
        state.begin(command.encode())
        self.assertEqual(state.lengths,[0]*4)
        self.assertEqual(token_rows(command),2)
        state.commit(command)
        self.assertEqual(state.lengths,[7,0,0,4])
        self.assertEqual(state.generations,[1,0,0,1])
        self.assertEqual(state.sequence,2)

    def test_invalid_budget_or_ownership_rejects_without_mutation(self):
        for rows,entries in ((9,((0,1,5),(1,1,4))),
                             (5,((0,1,2),(1,1,2))),
                             (4,((0,1,0),(1,1,4))),
                             (4,((0,1,2),(0,1,2))),
                             (4,((0,2,2),(1,1,2))),
                             (4,((0,1,4),))):
            state=StageState(1,8,4)
            with self.subTest(rows=rows,entries=entries),self.assertRaises(ValueError):
                state.begin(Command(1,1,PREFILL_BATCH,rows,entries).encode())
            self.assertEqual(state.lengths,[0]*4);self.assertEqual(state.sequence,1)
            self.assertIsNone(state.pending)
        state=StageState(1,8,4)
        state.commit(state.begin(Command(1,1,PREFILL,2,((0,1,0),)).encode()))
        with self.assertRaises(ValueError):state.begin(Command(1,2,PREFILL_BATCH,4,((0,2,2),(1,1,2))).encode())
        self.assertEqual(state.lengths,[2,0,0,0])


class ChunkProtocolTests(unittest.TestCase):
    def test_explicit_context_limit_and_chunk_generation_offsets(self):
        from damendaemon.stage_protocol import PREFILL_CHUNK
        with self.assertRaises(ValueError):StageState(1,513)
        for limit in (True,0,2**20+1):
            with self.assertRaises(ValueError):StageState(1,512,context_limit=limit)
        state=StageState(1,32768,context_limit=32768)
        first=Command(1,1,PREFILL_CHUNK,32767,((0,1,0),))
        self.assertEqual(token_rows(first),1);state.commit(state.begin(first.encode()))
        for entry in ((0,2,32767),(0,1,32766),(0,2,0)):
            with self.assertRaises(ValueError):state.begin(Command(1,2,PREFILL_CHUNK,1,(entry,)).encode())
        command=Command(1,2,PREFILL_CHUNK,1,((0,1,32767),));state.commit(state.begin(command.encode()))
        self.assertEqual(state.lengths[0],32768)
        with self.assertRaises(ValueError):state.begin(Command(1,3,PREFILL_CHUNK,1,((0,1,32768),)).encode())
        state.commit(state.begin(Command(1,3,RELEASE,0,((0,1,32768),)).encode()))
        state.commit(state.begin(Command(1,4,PREFILL_CHUNK,2,((0,2,0),)).encode()))
        self.assertEqual((state.lengths[0],state.generations[0]),(2,2))

    def test_previous_wire_version_cannot_silently_accept_chunk_protocol(self):
        from damendaemon.stage_protocol import VERSION,PREFILL_CHUNK
        words=Command(1,1,PREFILL_CHUNK,2,((0,1,0),)).encode();words[1]=VERSION-1
        with self.assertRaisesRegex(ValueError,'protocol'):decode(words)


class ModelSwitchProtocolTests(unittest.TestCase):
    def test_switch_commits_new_epoch_only_after_release_and_peer_ack(self):
        state=StageState(1,512,2,models=2)
        prompt=Command(1,1,PREFILL,8,((0,1,0),))
        state.commit(state.begin(prompt.encode()))
        with self.assertRaises(ValueError):state.begin(Command(1,2,SWITCH,1,()).encode())
        self.assertIsNone(state.pending)
        release=Command(1,2,RELEASE,0,((0,1,8),))
        state.commit(state.begin(release.encode()))
        switch=Command(1,3,SWITCH,1,())
        state.begin(switch.encode())
        self.assertEqual((state.model_index,state.epoch,state.sequence),(0,1,3))
        state.commit(switch)
        self.assertEqual((state.model_index,state.epoch,state.sequence),(1,2,1))
        self.assertEqual(state.generations,[0,0])
        with self.assertRaises(ValueError):state.begin(prompt.encode())
        with self.assertRaises(ValueError):state.begin(switch.encode())
        back=Command(2,1,SWITCH,0,())
        state.commit(state.begin(back.encode()))
        self.assertEqual((state.model_index,state.epoch,state.sequence),(0,3,1))

    def test_invalid_targets_fail_without_state_mutation(self):
        for models,target,entries in ((1,1,()),(2,2,()),(2,0,()),(2,1,((0,1,0),))):
            state=StageState(1,512,models=models)
            with self.subTest(models=models,target=target),self.assertRaises(ValueError):
                state.begin(Command(1,1,SWITCH,target,entries).encode())
            self.assertEqual((state.model_index,state.epoch,state.sequence),(0,1,1))
            self.assertIsNone(state.pending)

    def test_failed_switch_does_not_publish_target(self):
        state=StageState(1,512,models=2)
        command=state.begin(Command(1,1,SWITCH,1,()).encode())
        state.poison()
        with self.assertRaises(RuntimeError):state.commit(command)
        self.assertEqual((state.model_index,state.epoch),(0,1))
        self.assertTrue(state.failed)


class SlotHandoffProtocolTests(unittest.TestCase):
    def test_handoff_installs_populated_slot_as_fresh_generation(self):
        state=StageState(1,512)
        command=state.begin(Command(1,1,KV_HANDOFF,4,((0,1,4),)).encode())
        self.assertEqual(command.entries,((0,1,4),))
        state.commit(command)
        self.assertEqual((state.lengths,state.generations,state.sequence),([4,0,0,0],[1,0,0,0],2))
        # The installed slot decodes and chunk-continues like a prefilled one.
        decode=Command(1,2,DECODE,1,((0,1,4),))
        state.commit(state.begin(decode.encode()))
        self.assertEqual(decode.operation,DECODE)
        self.assertEqual(state.lengths[0],5)
        chunk=Command(1,3,PREFILL_CHUNK,2,((0,1,5),))
        state.commit(state.begin(chunk.encode()))
        self.assertEqual(chunk.operation,PREFILL_CHUNK)
        self.assertEqual(state.lengths[0],7)
        # A second handoff needs the NEXT generation (stale claims refuse).
        stale=Command(1,4,KV_HANDOFF,3,((0,1,3),))
        with self.assertRaises(ValueError):state.begin(stale.encode())
        state.commit(state.begin(Command(1,4,RELEASE,0,((0,1,7),)).encode()))
        state.commit(state.begin(Command(1,5,KV_HANDOFF,3,((0,2,3),)).encode()))
        self.assertEqual(state.lengths[0],3)

    def test_handoff_rejects_partial_occupied_or_out_of_range(self):
        for rows,entries in ((4,((0,0,4),)),(4,((0,2,4),)),(4,((0,1,3),)),
                             (0,((0,1,0),)),(4,((0,1,4),(1,1,4))),
                             (513,((0,1,513),))):
            state=StageState(1,512)
            with self.subTest(rows=rows,entries=entries),self.assertRaises(ValueError):
                state.begin(Command(1,1,KV_HANDOFF,rows,entries).encode())
            self.assertEqual((state.lengths,state.generations,state.sequence),([0]*4,[0]*4,1))
            self.assertIsNone(state.pending)
        # An occupied slot refuses a handoff (the fresh-slot rule of PREFILL).
        state=StageState(1,512)
        state.commit(state.begin(Command(1,1,PREFILL,2,((0,1,0),)).encode()))
        with self.assertRaises(ValueError):
            state.begin(Command(1,2,KV_HANDOFF,4,((0,2,4),)).encode())
        self.assertEqual(state.lengths[0],2)

    def test_previous_wire_version_cannot_silently_accept_handoff_protocol(self):
        from damendaemon.stage_protocol import VERSION
        words=Command(1,1,KV_HANDOFF,4,((0,1,4),)).encode();words[1]=VERSION-1
        with self.assertRaisesRegex(ValueError,'protocol'):decode(words)
