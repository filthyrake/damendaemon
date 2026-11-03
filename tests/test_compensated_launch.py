"""Shape/validation coverage for the compensated Q4 projection launch policy."""
import sys
import types
import unittest
from unittest import mock

# compensated_q8 imports triton at module scope (bare `import triton`), so a
# triton-less host needs the minimal stub present WHILE that module body
# executes. The stub must NOT persist in sys.modules past this import: a
# leaked fake triton makes every later test in the same pytest session
# dispatch on "triton available" and silently take the (Mock) fused path —
# #697: the 37-file-prefix pollution of test_hybrid_mtp_serving et al. Keep
# the stub alive only for this module's own reference; remove it from
# sys.modules so the rest of the session sees the honest ImportError and
# later modules take their own graceful no-triton fallbacks.
try:
    import triton  # noqa: F401
    _installed_stub = None
except ImportError:
    from unittest import mock as _mock
    triton = types.ModuleType('triton')
    language = types.ModuleType('triton.language')

    class _Kernel:
        def __init__(self, fn):
            self.fn = fn
            self.run = _mock.Mock()

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(grid, *args, **kwargs)

    triton.jit = lambda fn: _Kernel(fn)
    triton.cdiv = lambda a, b: -(-a // b)
    language.constexpr = object
    triton.language = language
    sys.modules['triton'] = triton
    sys.modules['triton.language'] = language
    _installed_stub = (triton, language)

import torch

from damendaemon import compensated_q8

if _installed_stub is not None:
    # The module body above executed under the stub (compensated_q8 holds its
    # own references); undo the sys.modules installation for the session.
    del sys.modules['triton']
    del sys.modules['triton.language']


class LaunchValidationTest(unittest.TestCase):
    def setUp(self):
        self.weight = torch.zeros((64, 128), dtype=torch.uint8)
        self.scales = torch.ones((64, 8), dtype=torch.float16)
        self.x = torch.zeros((127, 256), dtype=torch.float32)

    def linear(self, **kwargs):
        return compensated_q8.linear_q4(self.x, self.weight, self.scales, **kwargs)

    def launch(self, **kwargs):
        """Run against a mocked Triton launcher; returns the grid, args, launch kwargs."""
        with mock.patch.object(compensated_q8, '_ensure_cuda', lambda *items: None), \
                mock.patch.object(compensated_q8._project, 'run') as run:
            self.linear(**kwargs)
        if run.call_args is None:
            raise AssertionError('the kernel was never launched')
        positional = run.call_args.args
        grid = positional[0]
        args = positional[1:]
        launch_kwargs = run.call_args.kwargs
        return grid, args, launch_kwargs

    def test_row_tile_128_supported(self):
        grid, args, kwargs = self.launch(row_tile=128)
        self.assertEqual(kwargs['num_warps'], 4)
        self.assertEqual(kwargs['num_stages'], 2)
        self.assertEqual(args[-4], 128)  # BM constexpr (…, M, N, K, BM, BN, BK, Q4)

    def test_invalid_tiles_rejected(self):
        with self.assertRaises(ValueError):
            self.linear(row_tile=96)
        with self.assertRaises(ValueError):
            self.linear(reduction_tile=256)

    def test_warps_and_stages_forwarded(self):
        _, _, kwargs = self.launch(row_tile=64, reduction_tile=64, num_warps=8, num_stages=3)
        self.assertEqual(kwargs['num_warps'], 8)
        self.assertEqual(kwargs['num_stages'], 3)

    def test_grid_covers_all_rows_with_one_m_tile_at_128(self):
        grid, args, _ = self.launch(row_tile=128)
        rows = self.x.shape[0]
        self.assertEqual(grid[0], (rows + 127) // 128)
        self.assertEqual(grid[0], 1)

    def test_triton_stub_does_not_leak_into_the_session(self):
        """#697: on triton-less hosts the stub must not persist in sys.modules
        past this module's import — a leaked fake triton flips every later
        module's availability dispatch onto the Mock fused path."""
        if _installed_stub is None:
            self.skipTest('real triton present; the stub never installs')
        stub, language = _installed_stub
        self.assertNotIn('triton', sys.modules)
        self.assertNotIn('triton.language', sys.modules)
        # compensated_q8 keeps its own references — the module stays usable.
        self.assertIs(compensated_q8.tr, stub)


if __name__ == '__main__':
    unittest.main()
