"""Vision geometry admission: 0.8B and 2B enumerated, everything else rejected (#573/#575)."""
import json
import tempfile
import unittest
from pathlib import Path

from damendaemon.vision_policy import eligible, validate_model
from damendaemon.vision_qwen35 import read_vision_config, vision_gates

FIXTURE_CONFIG = Path('/Users/damenknight/.hermes/cache/scratch/qwen35-08b/config.json')


class GeometryAdmissionTests(unittest.TestCase):
    def test_both_qualified_text_geometries_are_eligible(self):
        for layers, hidden in ((24, 1024), (24, 2048), (64, 5120)):
            config = {'model_type': 'qwen3_5_text', 'hidden_size': hidden, 'num_hidden_layers': layers}
            self.assertTrue(eligible(config))
            validate_model(config, 1024)

    def test_other_geometries_stay_rejected(self):
        for config in ({'model_type': 'qwen3_5_text', 'hidden_size': 1536, 'num_hidden_layers': 24},
                       {'model_type': 'qwen3_5_text', 'hidden_size': 2048, 'num_hidden_layers': 40},
                       {'model_type': 'qwen3_5_text', 'hidden_size': 5120, 'num_hidden_layers': 24},
                       {'model_type': 'qwen3_5_text', 'hidden_size': 2048, 'num_hidden_layers': 64},
                       {'model_type': 'qwen3_5_moe', 'hidden_size': 2048, 'num_hidden_layers': 24},
                       {'model_type': 'qwen2', 'hidden_size': 1024, 'num_hidden_layers': 24}):
            self.assertFalse(eligible(config))
            with self.assertRaisesRegex(ValueError, 'qualified'):
                validate_model(config, 1024)

    def test_context_limit_applies_to_both(self):
        for layers, hidden in ((24, 1024), (24, 2048), (64, 5120)):
            with self.assertRaisesRegex(ValueError, 'context <=1024'):
                validate_model({'model_type': 'qwen3_5_text', 'hidden_size': hidden,
                                'num_hidden_layers': layers}, 1025)

    @unittest.skipUnless(FIXTURE_CONFIG.exists(), 'local 0.8B fixture config not staged')
    def test_vision_config_admits_both_encoder_geometries_and_rejects_others(self):
        base = json.loads(FIXTURE_CONFIG.read_text())
        for depth, hidden, heads, inner, out in ((12, 768, 12, 3072, 1024),
                                                 (24, 1024, 16, 4096, 2048),
                                                 (27, 1152, 16, 4304, 5120)):
            config = json.loads(json.dumps(base))
            config['vision_config'].update(depth=depth, hidden_size=hidden, num_heads=heads,
                                           intermediate_size=inner, out_hidden_size=out)
            directory = Path(tempfile.mkdtemp())
            (directory / 'config.json').write_text(json.dumps(config))
            self.assertEqual(read_vision_config(directory)['depth'], depth)
        config = json.loads(json.dumps(base))
        config['vision_config']['depth'] = 13
        directory = Path(tempfile.mkdtemp())
        (directory / 'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'unsupported Qwen3.5 vision geometry'):
            read_vision_config(directory)

    def test_gate_selection_is_per_depth(self):
        self.assertEqual(vision_gates(12)['relative_rms'], 1e-4)
        self.assertEqual(vision_gates(12)['max_absolute_element_relative'], 5e-4)
        self.assertEqual(vision_gates(24)['relative_rms'], 7e-4)
        self.assertEqual(vision_gates(24)['max_absolute_element_relative'], 2e-3)
        self.assertEqual(vision_gates(12)['max_absolute_floor'], 0.02)
        self.assertEqual(vision_gates(24)['max_absolute_floor'], 0.02)
        with self.assertRaisesRegex(ValueError, 'no calibrated vision gate'):
            vision_gates(18)

    def test_depth27_gate_carries_depth24_bounds_after_calibration(self):
        # PR #661: measured depth-27 oracle device spread (worst RMS
        # 1.26e-05, worst element-relative 4.2e-05) sits ~50x below the
        # depth-24-calibrated bounds, so they carry over unchanged.
        self.assertEqual(vision_gates(27), vision_gates(24))
        with self.assertRaisesRegex(ValueError, 'no calibrated vision gate'):
            vision_gates(28)


if __name__ == '__main__':
    unittest.main()
