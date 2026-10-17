import math
import unittest
from unittest.mock import patch

import torch

from damendaemon.hyper_connections import mixing_weights, reduce_head, reduce_streams, restore_streams


def scalar_weights(values, scale, bias, streams, iterations, epsilon):
    sigmoid = lambda x: 1/(1+math.exp(-x))
    pre = [sigmoid(values[i]*scale[0]+bias[i])+epsilon for i in range(streams)]
    post = [2*sigmoid(values[streams+i]*scale[1]+bias[streams+i]) for i in range(streams)]
    matrix = []
    for source in range(streams):
        row = [values[2*streams+source*streams+j]*scale[2]+bias[2*streams+source*streams+j]
               for j in range(streams)]
        exp = [math.exp(x-max(row)) for x in row]
        matrix.append([x/sum(exp)+epsilon for x in exp])
    for iteration in range(iterations):
        if iteration:
            matrix = [[x/(sum(row)+epsilon) for x in row] for row in matrix]
        sums = [sum(row[j] for row in matrix)+epsilon for j in range(streams)]
        matrix = [[x/sums[j] for j, x in enumerate(row)] for row in matrix]
    return pre, post, matrix


class HyperConnectionTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        rng = torch.Generator().manual_seed(73)
        hidden = torch.randn(2, 3, 4, 5, generator=rng).to(dtype)
        projection = torch.randn(24, 20, generator=rng)*.2
        return hidden, projection, torch.tensor([.7, .3, 1.2]), torch.randn(24, generator=rng)*.1

    def test_scalar_sinkhorn_reference(self):
        for streams in (1, 2, 4):
            values = torch.linspace(-2, 3, streams*(streams+2)).reshape(1, 1, -1)
            bias = torch.linspace(.3, -.2, values.numel())
            scale = torch.tensor([.7, 1.2, -.6])
            for iterations in (1, 3, 20):
                for epsilon in (1e-6, 1e-3):
                    expected = scalar_weights(values.flatten().tolist(), scale.tolist(), bias.tolist(),
                                              streams, iterations, epsilon)
                    actual = mixing_weights(values, scale, bias, streams,
                                            iterations=iterations, epsilon=epsilon)
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result[0, 0], torch.tensor(reference), rtol=2e-5, atol=1e-6)
                    if iterations == 20:
                        torch.testing.assert_close(actual[2].sum(-2), torch.ones(1, 1, streams), rtol=0, atol=2*epsilon)
                        torch.testing.assert_close(actual[2].sum(-1), torch.ones(1, 1, streams), rtol=0, atol=2*epsilon)

    def test_source_destination_orientation(self):
        result = restore_streams(torch.tensor([[[5., 6.]]]), torch.tensor([[[[1., 2.], [10., 20.]]]]),
                                 torch.tensor([[[.1, .2]]]), torch.tensor([[[[1., 2.], [3., 4.]]]]))
        torch.testing.assert_close(result, torch.tensor([[[[31.5, 62.6], [43., 85.2]]]]))

    def test_learned_reductions_against_scalar_reference(self):
        hidden, projection, scale, bias = self.fixture()
        actual, _, _ = reduce_streams(hidden, projection, scale, bias)
        head = reduce_head(hidden, projection[:4], scale[:1], bias[:4])
        for b in range(2):
            for s in range(3):
                rows = hidden[b, s].tolist()
                flat = [x for row in rows for x in row]
                norm = math.sqrt(sum(x*x for x in flat)/len(flat)+1e-6)
                mixes = [sum(x*w for x, w in zip(flat, row))/norm for row in projection.tolist()]
                pre, _, _ = scalar_weights(mixes, scale.tolist(), bias.tolist(), 4, 20, 1e-6)
                expected = [sum(pre[i]*rows[i][j] for i in range(4)) for j in range(5)]
                torch.testing.assert_close(actual[b, s], torch.tensor(expected), rtol=2e-5, atol=1e-6)
                weights = [1/(1+math.exp(-(mixes[i]*scale[0].item()+bias[i].item())))+1e-6 for i in range(4)]
                expected_head = [sum(weights[i]*rows[i][j] for i in range(4)) for j in range(5)]
                torch.testing.assert_close(head[b, s], torch.tensor(expected_head), rtol=2e-5, atol=1e-6)

    def test_storage_dtype_empty_reordering_and_no_host_extraction(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden, projection, scale, bias = self.fixture(dtype)
            before = hidden.clone()
            expected = reduce_streams(hidden, projection, scale, bias)
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                    patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                    patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host values')):
                actual = reduce_streams(hidden.flip(0), projection, scale, bias)
                restored = restore_streams(actual[0], hidden.flip(0), actual[1], actual[2])
                head = reduce_head(hidden, projection[:4], scale[:1], bias[:4])
                for empty in (hidden[:0], hidden[:, :0]):
                    out, post, matrix = reduce_streams(empty, projection, scale, bias)
                    self.assertEqual(restore_streams(out, empty, post, matrix).shape, empty.shape)
                    self.assertEqual(reduce_head(empty, projection[:4], scale[:1], bias[:4]).shape, empty.shape[:2]+(5,))
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference.flip(0))
            self.assertEqual(restored.dtype, dtype)
            self.assertEqual(head.dtype, dtype)
            self.assertEqual(actual[0].dtype, dtype)
            self.assertEqual(actual[1].dtype, torch.float32)
            self.assertEqual(actual[2].dtype, torch.float32)
            torch.testing.assert_close(hidden, before, rtol=0, atol=0)

    def test_invalid_parameters_before_projection(self):
        hidden, projection, scale, bias = self.fixture()
        with patch('damendaemon.hyper_connections.F.linear', side_effect=AssertionError('projection started')):
            for options in ({'iterations':0}, {'iterations':True}, {'iterations':129},
                            {'norm_epsilon':float('nan')}, {'connection_epsilon':0},
                            {'norm_epsilon':float('inf')}):
                with self.assertRaises(ValueError):
                    reduce_streams(hidden, projection, scale, bias, **options)
            for args in ((hidden, projection.half(), scale, bias), (hidden, projection[:, :-1], scale, bias),
                         (hidden, projection, scale[:1], bias), (hidden.int(), projection, scale, bias),
                         (hidden, projection.to('meta'), scale, bias)):
                with self.assertRaises(ValueError):
                    reduce_streams(*args)
            with self.assertRaises(ValueError):
                reduce_head(hidden, projection[:4], scale, bias[:4])
        out, post, matrix = reduce_streams(hidden, projection, scale, bias)
        for args in ((out.half(), hidden, post, matrix), (out, hidden, post.half(), matrix),
                     (out, hidden, post, matrix[..., :3]), (out[:, :1], hidden, post, matrix)):
            with self.assertRaises(ValueError):
                restore_streams(*args)
