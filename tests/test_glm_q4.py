"""GLM Q4_0 packing tests (#719): blockwise-FP8 decode, bank packing, executor admission."""
import unittest
from types import SimpleNamespace

import torch

from damendaemon.glm_q4 import (BLOCK, decode_blockwise_fp8, projected_q4_bank_bytes,
                                q4_bank_from_fp8, q4_expert, q4_matrix)
from damendaemon.packed_projection import packed_linear
from damendaemon.quantization import dequantize_q4, quantize_q4


def _pair(rows, width, seed=7):
    """Synthetic shipped-style blockwise-FP8 tensor with a partial tail grid.

    Codes avoid the E4M3FN NaN encodings (0x7F/0xFF) — real checkpoint
    weights are finite and the packer refuses non-finite truth.
    """
    rng = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (rows, width), generator=rng).to(torch.uint8)
    codes[codes == 0x7F] = 0x7E
    codes[codes == 0xFF] = 0xFE
    grid = torch.randn(-(-rows // BLOCK), -(-width // BLOCK), generator=rng).abs() + 0.1
    return codes, grid


class GlmQ4DecodeTests(unittest.TestCase):
    def test_decode_matches_manual_expansion_with_partial_tail(self):
        # 576 rows -> 4 full blocks + one 64-row tail block (the real kv_a shape)
        codes, grid = _pair(576, 6144)
        truth = decode_blockwise_fp8(codes, grid)
        # BYTE reinterpretation is the shipped semantic (`.to()` would be a
        # numeric cast and decode garbage — see glm_q4.decode_blockwise_fp8).
        expected = (codes.view(torch.float8_e4m3fn).float()
                    * grid.repeat_interleave(BLOCK, 0)[:576].repeat_interleave(BLOCK, 1)[:, :6144])
        torch.testing.assert_close(truth, expected, rtol=0, atol=0)
        self.assertEqual(truth.dtype, torch.float32)
        # The fp8 value decode must agree with the production tile decoder
        # bit-for-bit on every finite code (decode with unit scales = raw values).
        from damendaemon.packed_tiles import _e4m3
        unit = decode_blockwise_fp8(codes, torch.ones(-(-576 // BLOCK), -(-6144 // BLOCK)))
        self.assertTrue(torch.equal(unit, _e4m3(codes)))

    def test_decode_refuses_mismatched_grid_and_rank(self):
        codes, grid = _pair(128, 256)
        with self.assertRaises(ValueError):
            decode_blockwise_fp8(codes, grid[:1, :1])  # wrong grid
        with self.assertRaises(ValueError):
            decode_blockwise_fp8(codes[0], grid)  # 1-D codes
        with self.assertRaises(ValueError):
            decode_blockwise_fp8(codes, grid[0])  # 1-D grid


class GlmQ4PackTests(unittest.TestCase):
    def test_packed_layout_is_canonical_q4_0(self):
        codes, grid = _pair(64, 64)
        truth = decode_blockwise_fp8(codes, grid)
        matrix = q4_matrix(truth)
        self.assertEqual(matrix.format, 'q4_0_f16')
        self.assertEqual(matrix.weight.dtype, torch.uint8)
        self.assertEqual(tuple(matrix.weight.shape), (64, 32))
        self.assertEqual(matrix.scales.dtype, torch.float16)
        self.assertEqual(tuple(matrix.scales.shape), (64, 2))
        reference_weight, reference_scales = quantize_q4(truth)
        self.assertTrue(torch.equal(matrix.weight, reference_weight))
        self.assertTrue(torch.equal(matrix.scales, reference_scales))

    def test_packed_projection_equals_dequantized_reference(self):
        # packed_linear's q4_0_f16 tile decode must compute the same product
        # as dequantize_q4 @ x in FP32 (chunk order aside).
        codes, grid = _pair(32, 96)
        truth = decode_blockwise_fp8(codes, grid)
        matrix = q4_matrix(truth)
        hidden = torch.randn(5, 96, generator=torch.Generator().manual_seed(3))
        actual = packed_linear(hidden, matrix.weight, matrix.scales, format='q4_0_f16')
        expected = hidden @ dequantize_q4(matrix.weight, matrix.scales).t()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_expert_and_bank_packing_geometry(self):
        rng = torch.Generator().manual_seed(11)
        # gate/up (intermediate, width); down (width, intermediate); both dims
        # divisible by 32 (Q4_0 group requirement on the contraction side)
        triple = (torch.randn(32, 64, generator=rng), torch.randn(32, 64, generator=rng),
                  torch.randn(64, 32, generator=rng))
        expert = q4_expert(triple)
        self.assertEqual((expert.width, expert.intermediate), (64, 32))
        with self.assertRaises(ValueError):
            q4_expert((triple[0], triple[1]))  # not a triple
        with self.assertRaises(ValueError):
            q4_expert((triple[0][:, :33], triple[1][:, :33], triple[2][:33]))  # width not divisible by 32

        def pair3(seed):
            return (_pair(32, 64, seed), _pair(32, 64, seed + 1), _pair(64, 32, seed + 2))
        bank = q4_bank_from_fp8([pair3(21), pair3(31), pair3(41)])
        self.assertEqual(len(bank), 3)
        self.assertTrue(all(e.gate.format == 'q4_0_f16' for e in bank))
        self.assertTrue(all((e.width, e.intermediate) == (64, 32) for e in bank))
        with self.assertRaises(ValueError):
            q4_bank_from_fp8([(_pair(32, 64),)])  # entry is not a triple

    def test_projection_bytes_accounting(self):
        self.assertEqual(projected_q4_bank_bytes([(64, 32)]), int((64 * 32) * (0.5 + 2 / 32)))


class CudaGlmQ4AdmissionTests(unittest.TestCase):
    """CPU-visible admission pins: q4 bank admitted, mixed refused, default unchanged."""

    def _fake_expert(self, format):
        from damendaemon.cuda_packed_expert import CudaPackedExpert
        e = CudaPackedExpert.__new__(CudaPackedExpert)
        e.device = torch.device('cuda:0')
        e.width = 2
        e.workspace_bytes = 1024
        e.activation_limit = None
        e.gate = e.up = e.down = SimpleNamespace(format=format)
        return e

    def test_q4_bank_admitted_and_default_unchanged(self):
        from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward
        from torch._subclasses.fake_tensor import FakeTensorMode
        q4 = self._fake_expert('q4_0_f16')
        fp8 = self._fake_expert('e4m3fn_f32')
        with FakeTensorMode():
            CudaPackedGlmFeedForward(q4)  # admitted (dense path)
            CudaPackedGlmFeedForward(fp8)  # default unchanged
        # The sparse q4 bank with a real router is pinned on hardware by the
        # guarded A6000 cell (verify_glm_q4_a6000): router isfinite checks are
        # data-dependent and cannot execute on fake tensors.

    def test_mixed_or_unknown_format_refused(self):
        from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward
        from torch._subclasses.fake_tensor import FakeTensorMode
        q4 = self._fake_expert('q4_0_f16')
        fp8 = self._fake_expert('e4m3fn_f32')
        other = self._fake_expert('q8_0_f16')
        with FakeTensorMode():
            with self.assertRaises(ValueError):
                CudaPackedGlmFeedForward([q4, fp8], shared=q4, router=torch.ones(2, 2),
                                         correction_bias=torch.ones(2), workspace_bytes=1024)  # mixed bank
            with self.assertRaises(ValueError):
                CudaPackedGlmFeedForward(other, workspace_bytes=1024)  # not admitted
            mixed = self._fake_expert('q4_0_f16')
            mixed.up = SimpleNamespace(format='e4m3fn_f32')  # mixed within one expert
            with self.assertRaises(ValueError):
                CudaPackedGlmFeedForward(mixed, workspace_bytes=1024)


class GlmAttentionQ4ProjectionTests(unittest.TestCase):
    """#719 rung 4b: GLM attention projections admit canonical q4_0_f16 storage."""

    def _geometry(self, seed=5):
        # Tiny MLA geometry: dim=128, qr=64, kr=32, heads=4, nope width=32 —
        # every contraction side divisible by 32 (Q4_0 group requirement).
        rng = torch.Generator().manual_seed(seed)
        dim, qr, kr, heads, width = 128, 64, 32, 4, 32
        truth = {'q_a_proj.weight': torch.randn(qr, dim, generator=rng),
                 'q_b_proj.weight': torch.randn(heads * width, qr, generator=rng),
                 'kv_a_proj_with_mqa.weight': torch.randn(kr, dim, generator=rng)}
        truth['q_a_layernorm.weight'] = torch.ones(qr)
        truth['kv_a_layernorm.weight'] = torch.ones(kr)
        return truth, dict(dim=dim, qr=qr, kr=kr, heads=heads, width=width)

    def _packed(self, truth):
        from damendaemon.glm_attention import _projection
        from damendaemon.prepared_projection import PackedProjection
        packed = dict(truth)
        for name in ('q_a_proj.weight', 'q_b_proj.weight', 'kv_a_proj_with_mqa.weight'):
            weight, scales = quantize_q4(truth[name])
            packed[name] = PackedProjection(weight, scales, format='q4_0_f16')
            self.assertTrue(_projection(packed[name], 'cpu'))
        return packed

    def test_attention_projection_admits_q4_and_refuses_other_formats(self):
        from damendaemon.glm_attention import _projection
        from damendaemon.prepared_projection import PackedProjection
        truth, _ = self._geometry()
        packed = self._packed(truth)
        for name, value in packed.items():
            if 'layernorm' in name:
                continue
            self.assertTrue(_projection(value, 'cpu'))
        # A VALID q8_0_f16 projection is constructable but not admitted by
        # the GLM attention admission (e2m1/q8 stay out of the #719 surface).
        q8 = PackedProjection(torch.zeros((64, 128), dtype=torch.int8),
                              torch.ones((64, 4), dtype=torch.float16), format='q8_0_f16')
        self.assertFalse(_projection(q8, 'cpu'))
        self.assertFalse(_projection(packed['q_a_proj.weight'], 'cuda'))  # cpu tensor, cuda device

    def test_project_mla_executes_q4_projections(self):
        # Same _project_mla the CUDA flash path delegates to (rope_dim=0):
        # dense-FP32 truth vs q4-packed candidate through the identical code.
        from damendaemon.glm_attention import _project_mla
        truth, geometry = self._geometry()
        packed = self._packed(truth)
        rng = torch.Generator().manual_seed(9)
        hidden = torch.randn(1, 16, geometry['dim'], generator=rng)
        positions = torch.arange(16).reshape(1, 16)
        dense = _project_mla(hidden, positions, truth, heads=geometry['heads'],
                             nope_dim=geometry['width'], rope_dim=0)
        quantized = _project_mla(hidden, positions, packed, heads=geometry['heads'],
                                 nope_dim=geometry['width'], rope_dim=0)
        for name in ('query', 'low_rank', 'latents'):
            self.assertTrue(torch.isfinite(quantized[name]).all())
            # Synthetic Q4_0 drift tracks the real-weight envelope (rung 1:
            # 8-10% rel RMS; the standing #719 gate is 1.5x that family).
            rel_rms = (quantized[name] - dense[name]).norm() / dense[name].norm()
            self.assertLess(float(rel_rms), 0.15)


if __name__ == '__main__':
    unittest.main()
