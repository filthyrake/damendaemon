import unittest
import torch
from damendaemon.expert_routes import grouped_routes


class ExpertRouteTests(unittest.TestCase):
    def test_grouping_preserves_row_and_route_order_and_skips_empty_experts(self):
        for selected in (torch.tensor([[3,1,3],[0,3,1],[3,0,3]]),torch.zeros((7,4),dtype=torch.int64),
                         torch.tensor([[3,1],[1,3]]).t(),torch.empty((0,4),dtype=torch.int64)):
            actual=list(grouped_routes(selected,5));expected=[]
            for expert in range(5):
                tokens,routes=torch.where(selected==expert)
                if tokens.numel():expected.append((expert,tokens,tokens*selected.shape[1]+routes))
            self.assertEqual(len(actual),len(expected))
            for (ae,at,ai),(be,bt,bi) in zip(actual,expected):
                self.assertEqual(ae,be);self.assertTrue(torch.equal(at,bt));self.assertTrue(torch.equal(ai,bi))
            if actual:self.assertEqual(sorted(torch.cat([i for _,_,i in actual]).tolist()),list(range(selected.numel())))

    def test_bad_shape_dtype_and_bounds_fail(self):
        for selected,experts in ((torch.ones(4,dtype=torch.int64),4),(torch.ones(2,4),4),
                (torch.zeros(2,0,dtype=torch.int64),4),(torch.zeros(2,4,dtype=torch.int64),0),
                (torch.tensor([[0,4]]),4),(torch.tensor([[-1,0]]),4)):
            with self.subTest(selected=selected,experts=experts),self.assertRaises((ValueError,RuntimeError)):
                list(grouped_routes(selected,experts))
