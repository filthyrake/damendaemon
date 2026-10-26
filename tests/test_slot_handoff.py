"""Slot handoff tests (#606): payload identity checks and greedy-equal
continuation after a real handoff, on the CPU hybrid fixture (no GPU
qualification)."""
import unittest
from unittest.mock import patch

import torch

from damendaemon.hybrid import HybridExecutor
from damendaemon.hybrid_pipeline import HybridPipeline
from damendaemon.slot_handoff import export_slot, import_slot, release_slot
from tests import test_hybrid_stage


class SlotHandoffTests(test_hybrid_stage.HybridStageTests):
    def executor(self):
        with patch('damendaemon.hybrid.selected_device', return_value='cpu'), \
                patch('damendaemon.hybrid.empty_cache'), \
                patch('damendaemon.hybrid.available_memory', return_value=2**40):
            return HybridExecutor(self.model, use_graph=False, max_batch=2,
                                   kernels='torch')

    @staticmethod
    def _executor_chain(executor, prompt, steps):
        tokens = [int(executor.prefill(list(prompt)).argmax(-1)[0])]
        for _ in range(steps - 1):
            _, next_ids = executor.decode(tokens[-1])
            tokens.append(int(next_ids[0]))
        return tokens

    @staticmethod
    def _pipeline_chain(pipeline, prompt, steps):
        tokens = [int(pipeline.prefill(list(prompt)).argmax(-1)[0])]
        for _ in range(steps - 1):
            _, next_ids = pipeline.decode_batch([tokens[-1]], [0])
            tokens.append(int(next_ids[0]))
        return tokens

    @staticmethod
    def _handoff_chain(pipeline, first_token, steps):
        tokens = [first_token]
        for _ in range(steps - 1):
            _, next_ids = pipeline.decode_batch([tokens[-1]], [0])
            tokens.append(int(next_ids[0]))
        return tokens

    def test_handoff_greedy_tokens_match_sender_and_recomputed_pipeline(self):
        prompt = [1, 3, 5, 7]
        sender = self.executor()
        reference = self._executor_chain(sender, prompt, steps=8)
        # Disaggregated prefill contract: the sender's first sampled token
        # travels with the KV handoff (the receiver never re-runs the prompt).
        # The handoff payload is exported from a sender holding ONLY the
        # prompt prefix — exactly the state the reference chain started from.
        handoff_source = self.executor()
        handoff_source.prefill(prompt)
        payload = export_slot(handoff_source, 0)
        first = reference[0]
        stages = [self.stage(0, 2), self.stage(2, 4)]
        pipeline = HybridPipeline(stages)
        reports = pipeline.install_handoff(payload, slot=0, generation=1)
        handed = self._handoff_chain(pipeline, first, steps=8)
        self.assertEqual(handed, reference)
        # Control: a fresh pipeline recomputing the prompt agrees byte-for-byte.
        control = HybridPipeline([self.stage(0, 2), self.stage(2, 4)])
        self.assertEqual(self._pipeline_chain(control, prompt, steps=8), handed)
        # State equality: each rank holds exactly its layer slice of the
        # handoff source's populated slot.
        self.assertEqual([r['position'] for r in reports], [len(prompt)] * 2)
        self.assertEqual(pipeline.status()['handoffs'][0]['import_bytes'],
                         reports[0]['import_bytes'])
        for stage in pipeline.stages:
            for layer, cache in stage.kv.items():
                torch.testing.assert_close(cache[:, 0, :, :len(prompt)],
                                           handoff_source.kv[layer][:, 0, :, :len(prompt)],
                                           rtol=0, atol=0)
        # Sender release after the acknowledged handoff: clean slot, reusable.
        release_slot(handoff_source, 0)
        self.assertEqual(handoff_source.lengths[0], 0)

    def test_handoff_refuses_mismatched_receiver_and_occupied_slot(self):
        sender = self.executor()
        sender.prefill([1, 3, 5, 7])
        payload = export_slot(sender, 0)
        receiver = self.stage(0, 2)
        # Occupied target slot refuses before touching receiver state.
        receiver.lengths[0] = 2
        with self.assertRaises(ValueError):
            import_slot(receiver, 0, 1, payload)
        receiver.lengths[0] = 0
        # A tampered config / context / dtype refuses at install time.
        for key, value in (('config', {'different': 'config'}),
                           ('context', 31), ('compute_dtype', 'torch.float16'),
                           ('position', 0)):
            bad = {'metadata': {**payload['metadata'], key: value},
                   'tensors': payload['tensors']}
            with self.subTest(key=key), self.assertRaises(ValueError):
                import_slot(receiver, 0, 1, bad)
        # A receiver whose layer range the payload does not cover refuses.
        cropped = {'metadata': {**payload['metadata'], 'layer_end': 2},
                   'tensors': payload['tensors']}
        with self.assertRaises(ValueError):
            import_slot(self.stage(0, 4), 0, 1, cropped)
        # A payload missing an inventory tensor refuses before install.
        dropped = {'metadata': payload['metadata'],
                   'tensors': {name: tensor for name, tensor
                               in payload['tensors'].items()
                               if name != 'kv.data.1'}}
        with self.assertRaises(ValueError):
            import_slot(self.stage(0, 2), 0, 1, dropped)
        # An empty payload must not commit a cursor over nothing.
        with self.assertRaises(ValueError):
            import_slot(self.stage(0, 2), 0, 1,
                        {'metadata': {'config': payload['metadata']['config'],
                                      'quantization': 'none', 'kv_mode': 'none',
                                      'compute_dtype': payload['metadata']['compute_dtype'],
                                      'context': 32, 'max_batch': 2, 'position': 4,
                                      'layer_start': 0, 'layer_end': 4,
                                      'inventory': []},
                         'tensors': {}})

    def test_export_requires_populated_slot_and_records_inventory(self):
        sender = self.executor()
        with self.assertRaises(ValueError):
            export_slot(sender, 0)  # empty slot
        with self.assertRaises(ValueError):
            export_slot(sender, 99)  # out of range
        sender.prefill([2, 4, 6])
        payload = export_slot(sender, 0)
        metadata = payload['metadata']
        self.assertEqual(metadata['position'], 3)
        self.assertEqual(metadata['layer_start'], 0)
        self.assertEqual(metadata['layer_end'], 4)
        self.assertEqual(metadata['kv_mode'], getattr(sender, 'kv_mode', 'none'))
        self.assertEqual(metadata['quantization'], 'none')
        # Inventory covers kv (per layer), conv and state with byte counts.
        groups = {name.split('.')[0] for name, *_ in metadata['inventory']}
        self.assertEqual(groups, {'kv', 'conv', 'state'})
        for name, dtype, shape, byte_count in metadata['inventory']:
            tensor = payload['tensors'][name]
            self.assertEqual(list(tensor.shape), shape)
            self.assertEqual(byte_count, tensor.numel() * tensor.element_size())
        self.assertGreater(payload['telemetry']['export_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
