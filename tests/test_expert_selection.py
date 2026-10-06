import math
import unittest
from unittest.mock import patch

import torch

from dllm.expert_selection import select_experts


def oracle(row, top_k, scoring, normalize, groups, keep, bias, scale):
    if scoring == 'softmax':
        raw = [math.exp(value-max(row)) for value in row]
        probabilities = [value/sum(raw) for value in raw]
    else:
        probabilities = [1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value)) for value in row]
    scores = [value+(bias[i] if bias is not None else 0) for i, value in enumerate(probabilities)]
    width = len(row)//groups
    ranks = []
    for group in range(groups):
        ordered = sorted(scores[group*width:(group+1)*width], reverse=True)
        ranks.append(sum(ordered[:2]) if bias is not None else ordered[0])
    allowed = sorted(range(groups), key=lambda group: (-ranks[group], group))[:keep]
    candidates = [i for i in range(len(row)) if i//width in allowed]
    selected = sorted(candidates, key=lambda i: (-scores[i], i))[:top_k]
    weights = [probabilities[i] for i in selected]
    total = sum(weights)
    if normalize and total:
        weights = [value/total for value in weights]
    return [value*scale for value in weights], selected


class ExpertSelectionTests(unittest.TestCase):
    def test_grouped_selection_matches_independent_scalar_oracle(self):
        rows = [[-.2, 1.4, -1.1, .8, -.6, 2.1, -3., .4],
                [1.7, -.5, .3, 2.4, -1.9, .6, 1.1, -.1]]
        bias = [3., -2., -.8, .4, -1.5, -3., .7, .2]
        for scoring in ('softmax', 'sigmoid'):
            for correction in (None, bias):
                for normalize in (False, True):
                    for groups, keep, k in ((2, 1, 3), (4, 2, 3), (4, 4, 6)):
                        with self.subTest(scoring=scoring, bias=correction, normalize=normalize, groups=groups):
                            logits = torch.tensor(rows)
                            actual_weights, actual_ids = select_experts(logits, k, scoring=scoring,
                                renormalize=normalize, group_count=groups, groups_to_keep=keep,
                                correction_bias=None if correction is None else torch.tensor(correction), scale=1.7)
                            reference = [oracle(row, k, scoring, normalize, groups, keep, correction, 1.7) for row in rows]
                            self.assertEqual(actual_ids.tolist(), [r[1] for r in reference])
                            torch.testing.assert_close(actual_weights, torch.tensor([r[0] for r in reference]), rtol=1e-6, atol=1e-7)
                            self.assertEqual(actual_ids.dtype, torch.int64)

    def test_existing_softmax_dispatch_is_unchanged(self):
        logits = torch.randn(11, 64, generator=torch.Generator().manual_seed(7))
        for normalize in (False, True):
            expected, ids = logits.softmax(-1).topk(8, dim=-1)
            if normalize:
                expected = expected/expected.sum(-1, keepdim=True)
            actual, selected = select_experts(logits, 8, renormalize=normalize)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(selected, ids, rtol=0, atol=0)

    def test_bias_changes_selection_but_never_weights_and_excludes_negative_groups(self):
        logits = torch.tensor([[2., 1., -1., -2.]])
        bias = torch.tensor([-10., -10., -2., -2.])
        weights, ids = select_experts(logits, 2, scoring='sigmoid', group_count=2,
                                     groups_to_keep=1, correction_bias=bias)
        self.assertEqual(ids.tolist(), [[2, 3]])
        torch.testing.assert_close(weights, logits.sigmoid()[:, 2:])

    def test_ties_reordering_empty_batches_and_sigmoid_underflow(self):
        logits = torch.zeros(3, 8)
        kwargs = dict(group_count=4, groups_to_keep=2, correction_bias=torch.zeros(8))
        weights, ids = select_experts(logits, 3, **kwargs)
        self.assertEqual(ids.tolist(), [[0, 1, 2]]*3)
        reordered = select_experts(logits[[2, 0, 1]], 3, **kwargs)
        torch.testing.assert_close(reordered[0], weights[[2, 0, 1]])
        torch.testing.assert_close(reordered[1], ids[[2, 0, 1]])
        empty = select_experts(torch.empty(0, 8), 3, **kwargs)
        self.assertEqual(empty[0].shape, (0, 3))
        weights, _ = select_experts(torch.full((1, 8), -1000.), 2, scoring='sigmoid', renormalize=True)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights.tolist(), [[0., 0.]])

    def test_invalid_policies_are_rejected(self):
        logits = torch.ones(2, 8)
        for kwargs in ({'top_k':True}, {'top_k':9}, {'group_count':3},
                       {'group_count':4, 'groups_to_keep':1, 'top_k':3},
                       {'groups_to_keep':0}, {'scale':float('nan')}, {'scale':0},
                       {'renormalize':1}, {'scoring':'bad'}, {'correction_bias':torch.zeros(7)},
                       {'group_count':8, 'correction_bias':torch.zeros(8)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_experts(logits, **{'top_k':2, **kwargs})

    def test_selection_does_not_extract_routing_decisions_to_host(self):
        logits, bias = torch.randn(3, 8), torch.randn(8)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host copy')), \
                patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host routes')):
            weights, ids = select_experts(logits, 3, scoring='sigmoid', renormalize=True,
                                         group_count=4, groups_to_keep=2, correction_bias=bias)
        self.assertEqual(weights.shape, ids.shape)
