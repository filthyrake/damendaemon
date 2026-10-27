"""#4: hybrid Metal scan eligibility across the qualified geometry set."""
import unittest

from damendaemon.hybrid_metal import eligible

BASE = dict(model_type='qwen3_5_text', num_hidden_layers=24,
            linear_key_head_dim=128, linear_value_head_dim=128,
            linear_num_key_heads=16, linear_num_value_heads=16)


class HybridMetalEligibilityTests(unittest.TestCase):
    def test_qualified_geometries(self):
        # The 0.8B keeps its original 1024 bound (its own evidence); the 2B
        # carries this cell's 32K qualification.
        self.assertTrue(eligible({**BASE, 'hidden_size': 1024}, 1024))
        self.assertFalse(eligible({**BASE, 'hidden_size': 1024}, 1025))
        self.assertTrue(eligible({**BASE, 'hidden_size': 2048}, 1024))
        self.assertTrue(eligible({**BASE, 'hidden_size': 2048}, 32768))

    def test_context_bounds(self):
        # The 2B's own bound is 32K; the 0.8B's stays 1024.
        two_b = {**BASE, 'hidden_size': 2048}
        self.assertFalse(eligible(two_b, 0))
        self.assertTrue(eligible(two_b, 4096))
        self.assertTrue(eligible(two_b, 32768))
        self.assertFalse(eligible(two_b, 32769))
        self.assertFalse(eligible(two_b, 65536))

    def test_batch_bounds(self):
        config = {**BASE, 'hidden_size': 2048}
        self.assertTrue(eligible(config, 4096, max_batch=4))
        self.assertFalse(eligible(config, 4096, max_batch=5))
        self.assertFalse(eligible(config, 4096, max_batch=0))

    def test_unqualified_geometry_rejected(self):
        # hidden_size outside the qualified set (a future 7B-class backbone
        # with the same head layout is NOT admitted without new evidence).
        self.assertFalse(eligible({**BASE, 'hidden_size': 4096}, 1024))
        # head-layout changes break the kernel contract.
        self.assertFalse(eligible({**BASE, 'hidden_size': 2048,
                                   'linear_num_key_heads': 8}, 1024))
        self.assertFalse(eligible({**BASE, 'hidden_size': 2048,
                                   'linear_key_head_dim': 64}, 1024))
        self.assertFalse(eligible({**BASE, 'hidden_size': 2048,
                                   'num_hidden_layers': 48}, 1024))
        # dense or other hybrid families are not the scan's contract.
        self.assertFalse(eligible({**BASE, 'hidden_size': 2048,
                                   'model_type': 'qwen3'}, 1024))


if __name__ == '__main__':
    unittest.main()
