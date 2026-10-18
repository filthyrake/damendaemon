"""CPU-only admission/lifecycle checks; these do not qualify Intel execution."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from damendaemon import device, models, xpu_policy
from damendaemon.cli import main
from damendaemon.runtime import Runtime, require_gpu
from tests import test_config

ENV = {'DAMENDAEMON_DEVICE': 'xpu', 'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
       'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu'}


class XpuTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def api(self):
        return SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda *args: 'Intel(R) Arc(TM) B580 Graphics', set_device=Mock(),
            synchronize=Mock(), empty_cache=Mock(), mem_get_info=lambda: (5*1024**3, 12*1024**3),
            is_initialized=lambda: True, memory_allocated=lambda: 123,
            memory_reserved=lambda: 456, max_memory_allocated=lambda: 789)

    def test_lifecycle_and_status_never_touch_cuda(self):
        # No CUDA attribute: an accidental CUDA call fails the regression.
        api = self.api()
        with patch.dict('sys.modules', torch=SimpleNamespace(xpu=api)):
            require_gpu()
            api.set_device.assert_called_once_with(0)
            self.assertEqual(device.selected_device(), 'xpu')
            device.synchronize()
            device.empty_cache()
            api.synchronize.assert_called_once_with()
            api.empty_cache.assert_called_once_with()
            self.assertEqual(device.available_memory(), 3*1024**3)
            self.assertIn('B580', device.device_name())
            import threading
            runtime = Runtime.__new__(Runtime)
            runtime.registry_lock = threading.RLock()
            runtime.models = runtime.host = runtime.gpu = {}
            runtime.failed_builds = []
            runtime.failed_executors = []
            runtime.closing = runtime.closed = False
            runtime.context = 512
            runtime.host_bytes = 1024
            runtime.gpu_limit = runtime.host_limit = 1
            status = runtime._status()
            self.assertEqual(status['xpu_allocated_bytes'], 123)
            self.assertEqual(status['xpu_reserved_bytes'], 456)
            self.assertEqual(status['xpu_peak_allocated_bytes'], 789)
            self.assertIsNone(status['cuda_allocated_bytes'])
            self.assertIsNone(status['hip_allocated_bytes'])

    def test_runtime_requires_real_single_target_device(self):
        for change in ({'is_available': lambda: False}, {'device_count': lambda: 2},
                       {'get_device_name': lambda *args: 'Other GPU'}):
            api = self.api()
            for key, value in change.items():
                setattr(api, key, value)
            with patch.dict('sys.modules', torch=SimpleNamespace(xpu=api)), self.assertRaises(RuntimeError):
                require_gpu()
            api.set_device.assert_not_called()
        with patch.dict('sys.modules', torch=SimpleNamespace()), self.assertRaises(RuntimeError):
            require_gpu()

    def test_display_reserve_and_invalid_telemetry(self):
        for free, total, expected in ((1024, 2048, 0), (3*1024**3, 12*1024**3, 1024**3)):
            api = SimpleNamespace(mem_get_info=lambda: (free, total))
            self.assertEqual(xpu_policy.available_memory(SimpleNamespace(xpu=api)), expected)
        for values in ((-1, 100), (101, 100), (0, 0), (True, 10), (1.5, 10)):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                xpu_policy.available_memory(SimpleNamespace(xpu=SimpleNamespace(mem_get_info=lambda: values)))

    def test_opt_in_and_unsupported_features_rejected(self):
        for key, value in (('DAMENDAEMON_EXPERIMENTAL_XPU', '0'), ('ONEAPI_DEVICE_SELECTOR', 'opencl:cpu'),
                           ('DAMENDAEMON_KERNELS', 'triton'), ('DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT', '12'),
                           ('DAMENDAEMON_KV_CACHE_TOKENS', '512'), ('DAMENDAEMON_SPECULATIVE_TOKENS', '4'),
                           ('DAMENDAEMON_PREFILL_GRAPHS', '1'), ('DAMENDAEMON_EXPERIMENTAL_VISION', '1')):
            with self.subTest(key=key), patch.dict('os.environ', {key: value}), self.assertRaises(ValueError):
                xpu_policy.validate_environment()

    def test_model_rejection_precedes_tensor_and_tokenizer_loading(self):
        config = test_config.ConfigTests().config()
        xpu_policy.validate_model(config, 'none', 512, 4)
        xpu_policy.validate_model(config, 'q8_0', 512, 4)
        xpu_policy.validate_model(config, 'q8_0', 16384, 1)
        hybrid = test_config.ConfigTests().config(hidden_size=1024, num_hidden_layers=24,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            intermediate_size=3072, vocab_size=151936, model_type='qwen3_5_text',
            linear_key_head_dim=128, linear_value_head_dim=128,
            linear_num_key_heads=16, linear_num_value_heads=16, max_position_embeddings=40960)
        xpu_policy.validate_model(hybrid, 'q8_0', 1024, 4)
        for changes, mode, context, batch in (({'model_type':'qwen3_moe'}, 'none', 512, 1),
                ({'model_type':'qwen3_5_text'}, 'q8_0', 2048, 1),
                ({'model_type':'qwen3_5_text'}, 'q8_0', 1024, 5),
                ({}, 'q8_0', 16385, 1), ({'model_type':'qwen3_5_text'}, 'q8_0', 2048, 1),
                ({}, 'q4_0', 512, 1), ({'rope_scaling': {'rope_type': 'yarn'}}, 'q8_0', 512, 1),
                ({}, 'none', 4096, 1), ({}, 'none', 512, 5),
                ({'rope_scaling': {'rope_type': 'yarn'}}, 'none', 512, 1)):
            with self.assertRaises(ValueError):
                xpu_policy.validate_model({**config, **changes}, mode, context, batch)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text(json.dumps(config))
            # No tokenizer or weight files exist: rejection must precede either read.
            for call in (lambda: models.load(path, 4096), lambda: models.weight_bytes(path, 'q4_0')):
                with self.assertRaisesRegex(ValueError, 'Intel'):
                    call()
            from damendaemon.qwen2 import Executor
            model = SimpleNamespace(config=config, quantization='q4_0', max_context=512)
            with self.assertRaisesRegex(ValueError, 'Intel'):
                Executor(model)

    def test_context_limits_are_quantization_bounded(self):
        self.assertEqual(xpu_policy.context_limit('none'), 2048)
        self.assertEqual(xpu_policy.context_limit('q8_0'), 16384)
        self.assertEqual(xpu_policy.context_limit('q4_0'), 2048)
        xpu_policy.validate_limits(2048, 4, 'none')
        xpu_policy.validate_limits(16384, 1, 'q8_0')
        for mode, context, batch in (('none', 2049, 1), ('q8_0', 16385, 1), ('q8_0', 8192, 5),
                                     ('none', 2048, 5), ('q8_0', True, 1)):
            with self.subTest(mode=mode, context=context, batch=batch), self.assertRaises(ValueError):
                xpu_policy.validate_limits(context, batch, mode)

    def test_cli_rejects_missing_opt_in_before_download(self):
        errors = io.StringIO()
        with patch.dict('os.environ', DAMENDAEMON_EXPERIMENTAL_XPU='0'), \
                patch('sys.argv', ['damendaemon', 'run', 'Qwen/not-downloaded', '--prompt', 'hi', '--device', 'xpu']), \
                patch('sys.stderr', errors), patch('damendaemon.cli.resolve_model') as resolve:
            with self.assertRaises(SystemExit):
                main()
            resolve.assert_not_called()
            self.assertIn('DAMENDAEMON_EXPERIMENTAL_XPU', errors.getvalue())

    def test_executor_budgets_fp32_before_any_xpu_transfer(self):
        import torch
        from damendaemon.config import decoder_config
        from damendaemon.qwen2 import Executor
        config = decoder_config(test_config.ConfigTests().config(hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, vocab_size=64))
        model = SimpleNamespace(config=config, quantization='none', max_context=32,
                                weights={'embed': torch.zeros(64, 32, dtype=torch.float16)}, scales={})
        expected = 64*32*4 + 2*1*32*1*16*4 + 512*1024**2
        # One byte short fails before Tensor.to(xpu), even with auto kernels.
        with patch('damendaemon.qwen2.empty_cache'), patch('damendaemon.qwen2.available_memory', return_value=expected-1), \
                patch('torch.cuda.get_device_name', side_effect=AssertionError('CUDA touched')):
            executor = Executor.__new__(Executor)
            with self.assertRaises(MemoryError):
                executor.__init__(model)
            self.assertEqual(executor.required_memory_bytes, expected)
            self.assertEqual(executor.compute_dtype, torch.float32)
            self.assertIsNone(executor.kernels)

    def test_executor_budgets_q8_packed_storage_and_dequant_workspace(self):
        import torch
        from damendaemon.config import decoder_config
        from damendaemon.qwen2 import Executor
        config = decoder_config(test_config.ConfigTests().config(hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, vocab_size=64))
        up = torch.zeros(64, 64, dtype=torch.int8)
        model = SimpleNamespace(config=config, quantization='q8_0', max_context=32,
                                weights={'embed': torch.zeros(64, 32, dtype=torch.float16),
                                         '0.up': up},
                                scales={'0.up': torch.zeros(64, 2, dtype=torch.float16)})
        expected = (64*32*4          # embed promotes to FP32 on device
                    + 64*64          # packed int8 stays packed on device
                    + 64*2*2         # FP16 scales
                    + 2*1*32*1*16*4  # FP32 KV cache
                    + 64*2*32*4      # largest transient dequantized FP32 matrix
                    + 512*1024**2)
        with patch('damendaemon.qwen2.empty_cache'), patch('damendaemon.qwen2.available_memory', return_value=expected-1), \
                patch('torch.cuda.get_device_name', side_effect=AssertionError('CUDA touched')):
            executor = Executor.__new__(Executor)
            with self.assertRaises(MemoryError):
                executor.__init__(model)
            self.assertEqual(executor.required_memory_bytes, expected)
            self.assertEqual(executor.packed_workspace_bytes, 64*2*32*4)

    def test_eager_q8_linear_matches_dequantized_reference(self):
        import torch
        from damendaemon.quantization import dequantize, quantize
        from damendaemon.quantized_torch import linear
        generator = torch.Generator().manual_seed(63)
        weight = torch.randn(48, 64, generator=generator)
        data, scales = quantize(weight)
        x = torch.randn(5, 3, 64, generator=generator)
        expected = torch.nn.functional.linear(x, dequantize(data, scales, x.dtype))
        result = linear(x, data, scales)
        self.assertEqual(result.dtype, x.dtype)
        self.assertTrue(torch.equal(result, expected))

    def test_quant_linear_dispatches_xpu_to_eager_path(self):
        import torch
        from unittest.mock import PropertyMock
        from damendaemon.quantization import Linear
        weight = torch.zeros(16, 32, dtype=torch.int8)
        scales = torch.zeros(16, 1, dtype=torch.float16)
        x = torch.zeros(2, 32)
        fallback = Mock(side_effect=AssertionError('fallback used'))
        linear = Linear({'w': weight}, {'w': scales}, 'cpu', fallback)
        with patch('torch.Tensor.device', new_callable=PropertyMock,
                   return_value=torch.device('xpu')), \
                patch('damendaemon.quantized_torch.linear', return_value=torch.ones(2, 16)) as eager:
            result = linear(x, weight, None)
        eager.assert_called_once_with(x, weight, scales, None)
        fallback.assert_not_called()
        self.assertTrue(torch.equal(result, torch.ones(2, 16)))

    def test_invalid_runtime_capacity_rejects_before_device_initialization(self):
        for options in ({'context': 2049}, {'batch_size': 5}, {'batch_size': True}):
            with patch('damendaemon.runtime.require_gpu') as initialize:
                with self.assertRaisesRegex(ValueError, 'Intel'):
                    Runtime({}, **options)
                initialize.assert_not_called()
