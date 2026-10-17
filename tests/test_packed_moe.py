import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from damendaemon.packed_moe import PackedExpert, PackedMatrix, packed_moe
from tests.test_packed_projection import fixture


def expert(format, width=32, intermediate=64):
    matrices,dense = [],[]
    for rows,columns in ((intermediate,width),(intermediate,width),(width,intermediate)):
        _,raw,scales,values = fixture(format,rows,columns)
        # Keep nonlinear tests away from overflow; packed exponent bytes shift
        # by the same amount as the independent scalar-decoded control.
        scales = scales-5
        matrices.append(PackedMatrix(raw,scales,format))
        dense.append(values/32)
    return PackedExpert(*matrices,row_chunk=13,column_chunk=17,activation_limit=2),dense


def dense_expert(hidden, matrices, coefficient=None):
    gate,up,down = matrices
    gated = (hidden.float()@gate.T).to(hidden.dtype).float().clamp(max=2)
    raised = (hidden.float()@up.T).to(hidden.dtype).float().clamp(-2,2)
    activated = F.silu(gated)*raised
    if coefficient is not None:
        activated *= coefficient
    return (activated.to(hidden.dtype).float()@down.T).to(hidden.dtype)


class PackedMoETests(unittest.TestCase):
    def test_mixed_formats_shared_duplicate_routes_and_low_precision(self):
        first,a = expert('e2m1')
        second,b = expert('e4m3fn')
        shared,c = expert('e4m3fn',intermediate=32)
        rng = torch.Generator().manual_seed(157)
        original = torch.randn(3,32,generator=rng)*.2
        routes = torch.tensor([[0,1],[1,1],[1,0]])
        coefficients = torch.tensor([[.25,.75],[.4,.1],[0.,1.]])
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            hidden = original.to(dtype)
            actual = packed_moe(hidden,[first,second],routes,coefficients,shared=shared)
            expected = torch.zeros_like(hidden,dtype=torch.float32)
            for token in range(3):
                for route in range(2):
                    expected[token:token+1] += dense_expert(hidden[token:token+1],(a,b)[routes[token,route]],
                                                          coefficients[token,route]).float()
                expected[token:token+1] += dense_expert(hidden[token:token+1],c).float()
            torch.testing.assert_close(actual,expected.to(dtype),rtol=2e-4 if dtype == torch.float32 else .02,atol=1e-5 if dtype == torch.float32 else .002)
            order = torch.tensor([2,0,0])
            torch.testing.assert_close(packed_moe(hidden[order],[first,second],routes[order],coefficients[order],shared=shared),actual[order])
            torch.testing.assert_close(first.forward(hidden,route_weights=coefficients[:,:1]),
                                       dense_expert(hidden,a,coefficients[:,:1]),rtol=.02,atol=.002)

    def test_empty_and_invalid_routes_fail_before_projection(self):
        prepared,_ = expert('e2m1')
        hidden = torch.zeros(2,32)
        routes = torch.zeros(2,1,dtype=torch.int64)
        coefficients = torch.ones(2,1)
        with patch.object(prepared,'_project',side_effect=AssertionError('projection started')):
            self.assertEqual(packed_moe(hidden[:0],[prepared],routes[:0],coefficients[:0]).shape,(0,32))
            self.assertEqual(prepared.forward(hidden[:0]).shape,(0,32))
            for bad in (torch.tensor([[0],[-1]]),torch.tensor([[0],[1]]),routes.float(),routes.to('meta')):
                with self.assertRaises(ValueError):
                    packed_moe(hidden,[prepared],bad,coefficients)
            with self.assertRaises(ValueError):
                packed_moe(hidden,[prepared],routes,coefficients.double())
            with self.assertRaises(ValueError):
                prepared.forward(hidden,route_weights=coefficients[:,0])

    def test_preparation_rejects_geometry_and_budget_without_decoding(self):
        prepared,_ = expert('e2m1')
        with patch('damendaemon.packed_projection.decode_tile',side_effect=AssertionError('decode started')):
            for options in ({'workspace_bytes':1},{'activation_limit':float('nan')},{'row_chunk':True}):
                with self.assertRaises(ValueError):
                    PackedExpert(prepared.gate,prepared.up,prepared.down,**options)
            with self.assertRaises(ValueError):
                PackedExpert(prepared.gate,prepared.down,prepared.up)

    def test_input_storage_preserved_and_one_row_at_a_time(self):
        prepared,_ = expert('e2m1')
        hidden = torch.randn(4,32)
        snapshots = [(p.weight.clone(),p.scales.clone()) for p in (prepared.gate,prepared.up,prepared.down)]
        original = prepared._project
        def bounded(source,matrix):
            self.assertEqual(source.shape[0],1)
            return original(source,matrix)
        with patch.object(prepared,'_project',side_effect=bounded):
            packed_moe(hidden,[prepared],torch.zeros(4,1,dtype=torch.int64),torch.ones(4,1))
        for matrix,(weight,scales) in zip((prepared.gate,prepared.up,prepared.down),snapshots):
            torch.testing.assert_close(matrix.weight,weight)
            torch.testing.assert_close(matrix.scales,scales)
