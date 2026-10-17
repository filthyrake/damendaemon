"""CPU-only expanded prompt accounting before image patch allocations."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from damendaemon.vision_model import ImagePrompt, VisionModelWeights
from damendaemon.vision_messages import ImageMessages


class Text:
    config = dict(model_type='qwen3_5_text', hidden_size=1024, num_hidden_layers=24)
    max_context = 1024
    bytes = 1

    def __init__(self, text_tokens=1, count=1):
        self.tokens = [7] * text_tokens + [248053, 248056, 248054] * count

    def chat_tokens(self, messages, **options):
        return list(self.tokens)


class VisionPromptTests(unittest.TestCase):
    def model(self, text_tokens=1, count=1):
        return VisionModelWeights(Text(text_tokens, count), SimpleNamespace(bytes=1))

    def decoded(self, count=1):
        return ImageMessages([dict(role='user', content='images')],
            tuple(torch.zeros(8,8,3,dtype=torch.uint8) for _ in range(count)),
            tuple(str(i) for i in range(count)))

    def test_overflow_rejected_before_resize_or_patch_construction(self):
        for text_tokens, count in ((1000,1), (900,2)):
            with self.subTest(text_tokens=text_tokens, count=count), \
                    patch('damendaemon.vision_model.decode_image_messages', return_value=self.decoded(count)), \
                    patch('damendaemon.vision_model.prepare_images', side_effect=AssertionError('patch preparation reached')):
                with self.assertRaisesRegex(ValueError, 'expanded image prompt.*context'):
                    self.model(text_tokens,count).chat_tokens([])

    def test_valid_prompt_preserves_grid_pixel_identity_and_order(self):
        decoded = self.decoded(2)
        before = tuple(image.clone() for image in decoded.images)
        with patch('damendaemon.vision_model.decode_image_messages', return_value=decoded):
            result = self.model(1,2).chat_tokens([])
        self.assertIsInstance(result, ImagePrompt)
        self.assertEqual(len(result), 1 + 2 * (64 + 2))
        self.assertEqual(result, [7] + ([248053] + [248056]*64 + [248054])*2)
        self.assertEqual(result.grids, [[1,16,16],[1,16,16]])
        self.assertEqual(tuple(result.pixels.shape), (512,1536))
        self.assertTrue(bool((result.pixels == -1).all()))
        self.assertEqual(result.identities, ('0','1'))
        for image, original in zip(decoded.images,before):
            self.assertTrue(torch.equal(image,original))

    def test_text_only_bypasses_image_planning(self):
        model = self.model()
        with patch('damendaemon.vision_model.decode_image_messages', return_value=ImageMessages([],(),())), \
                patch('damendaemon.vision_model.prepare_images') as prepare:
            self.assertEqual(model.chat_tokens([]), model.text.tokens)
            prepare.assert_not_called()
