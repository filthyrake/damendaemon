"""Image admission ordering with CPU metadata; never construct an accelerator."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from damendaemon.vision_hybrid import VisionHybridExecutor
from damendaemon.vision_model import ImagePrompt


class ImageExecutorPreflightTests(unittest.TestCase):
    def executor(self):
        executor = object.__new__(VisionHybridExecutor)
        executor.context = 1024
        executor.max_batch = 4
        executor.vision = SimpleNamespace(encode=Mock(side_effect=AssertionError('encoder invoked')))
        return executor

    def invalid(self):
        return [([248056], [[1,4,4]]), ([248056], []), ([7], [[1,2,2]]),
                ([248056], [[1,3,2]]), ([-1,248056], [[1,2,2]]),
                ([True,248056], [[1,2,2]])]

    def test_invalid_geometry_never_reaches_encoder(self):
        for tokens, grids in self.invalid():
            executor = self.executor()
            with self.subTest(tokens=tokens,grids=grids), self.assertRaises(ValueError):
                executor.prefill(ImagePrompt(tokens,None,grids,()))
            executor.vision.encode.assert_not_called()

    def test_later_invalid_image_cannot_partially_dispatch_batch(self):
        for tokens, grids in self.invalid():
            executor = self.executor()
            executor.prefill = Mock(return_value=torch.zeros(1,2))
            with self.subTest(tokens=tokens,grids=grids), self.assertRaises(ValueError):
                executor.prefill_batch([[7],ImagePrompt(tokens,None,grids,())],[0,1])
            executor.prefill.assert_not_called()
            executor.vision.encode.assert_not_called()

    def test_valid_image_and_mixed_batch_dispatch_unchanged(self):
        executor = self.executor()
        features = torch.ones(1,3)
        executor.vision.encode = Mock(return_value=features)
        result = torch.ones(1,2)
        executor.prefill_images = Mock(return_value=result)
        prompt = ImagePrompt([7,248056],torch.zeros(4,1536),[[1,2,2]],('image',))
        self.assertIs(executor.prefill(prompt,2), result)
        executor.vision.encode.assert_called_once_with(prompt.pixels,prompt.grids)
        executor.prefill_images.assert_called_once_with(prompt,features,prompt.grids,2)
        executor.prefill = Mock(return_value=result)
        output = executor.prefill_batch([[7],prompt],[0,1])
        self.assertTrue(torch.equal(output,torch.ones(2,2)))
        self.assertEqual(executor.prefill.call_count,2)

    def test_direct_features_keep_geometry_and_exact_slot_gate(self):
        for tokens, grids in self.invalid():
            with self.subTest(tokens=tokens,grids=grids), self.assertRaises(ValueError):
                self.executor().prefill_images(tokens,None,grids)
        for slot in (True,1.0,-1,4):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                self.executor().prefill_images([248056],None,[[1,2,2]],slot)
