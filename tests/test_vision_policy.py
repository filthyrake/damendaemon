"""Serving admission stays restricted while explicit backend validation expands."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from damendaemon.runtime import make_executor


class VisionPolicyTests(unittest.TestCase):
    def test_oversized_apple_batch_cannot_enter_serving_executor(self):
        model=SimpleNamespace(config={'model_type':'qwen3_5_text'},max_context=1024,
                              vision_weights=object(),scales={"quantized":object()})
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'mps'}), \
                patch('torch.backends.mps.is_available',return_value=True), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            with self.assertRaisesRegex(ValueError,'at most four slots'):
                make_executor(model,max_batch=5)
            constructor.assert_not_called()


if __name__=='__main__':unittest.main()
