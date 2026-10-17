import math
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_hyper import FlashHyperConnection


def component(**options):
    torch.manual_seed(725)
    return FlashHyperConnection(torch.randn(8,6)*.2,torch.randn(8)*.1,torch.tensor([.4,.7,.3]),streams=2,**options)


def scalar_control(hidden,c):
    # Float64 Python scalar equations; no production mixing or restoration calls.
    pre_all=[];post_all=[];matrix_all=[];collapsed=[]
    h=c.streams;d=c.width
    for batch in hidden.tolist():
        bp=[];bo=[];bm=[];bc=[]
        for state in batch:
            flat=[x for row in state for x in row];inv=1/math.sqrt(sum(x*x for x in flat)/len(flat)+c.norm_epsilon)
            mix=[sum(w*x*inv for w,x in zip(row,flat)) for row in c.projection.tolist()]
            sigmoid=lambda x:1/(1+math.exp(-x))
            pre=[sigmoid(mix[i]*float(c.scale[0])+float(c.base[i]))+c.connection_epsilon for i in range(h)]
            post=[2*sigmoid(mix[h+i]*float(c.scale[1])+float(c.base[h+i])) for i in range(h)]
            logits=[[mix[2*h+i*h+j]*float(c.scale[2])+float(c.base[2*h+i*h+j]) for j in range(h)] for i in range(h)]
            matrix=[]
            for row in logits:
                exp=[math.exp(x-max(row)) for x in row];matrix.append([x/sum(exp)+c.connection_epsilon for x in exp])
            for iteration in range(c.iterations):
                if iteration:matrix=[[x/(sum(row)+c.connection_epsilon) for x in row] for row in matrix]
                sums=[sum(matrix[i][j] for i in range(h))+c.connection_epsilon for j in range(h)]
                matrix=[[x/sums[j] for j,x in enumerate(row)] for row in matrix]
            bp.append(pre);bo.append(post);bm.append(matrix);bc.append([sum(pre[i]*state[i][j] for i in range(h)) for j in range(d)])
        pre_all.append(bp);post_all.append(bo);matrix_all.append(bm);collapsed.append(bc)
    return tuple(torch.tensor(x) for x in (collapsed,post_all,matrix_all))


class FlashHyperTests(unittest.TestCase):
    def test_scalar_control_and_noncontiguous_inputs(self):
        for iterations in (1,20):
            c=component(iterations=iterations);hidden=torch.randn(2,3,3,2).transpose(-1,-2)
            original=hidden.clone()
            for actual,expected in zip(c.reduce(hidden),scalar_control(hidden,c)):
                torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-6)
            torch.testing.assert_close(hidden,original)

    def test_restore_source_destination_orientation_and_mean_head(self):
        c=component();state=torch.arange(12,dtype=torch.float32).reshape(1,2,2,3)
        post=torch.tensor([[[.2,1.5],[.7,.8]]]);matrix=torch.tensor([[[[.1,.9],[.3,.7]],[[.4,.6],[.2,.8]]]])
        output=torch.tensor([[[3.,7.,2.],[6.,1.,4.]]]);expected=torch.empty_like(state)
        for s in range(2):
            for destination in range(2):
                expected[0,s,destination]=post[0,s,destination]*output[0,s]+sum(matrix[0,s,source,destination]*state[0,s,source] for source in range(2))
        torch.testing.assert_close(c.restore(output,state,post,matrix),expected)
        torch.testing.assert_close(c.head(state),(state[:,:,0]+state[:,:,1])/2)
        torch.testing.assert_close(c.reduce(torch.zeros_like(state))[0],torch.zeros(1,2,3))

    def test_preallocation_budgets_for_all_operations(self):
        state=torch.zeros(1,2,2,3)
        for option in ('max_workspace_bytes','max_output_bytes'):
            c=component(**{option:1})
            with patch('damendaemon.glm_flash_hyper._finite',side_effect=AssertionError('scanned before admission')):
                for invoke in (lambda:c.reduce(state),lambda:c.head(state),lambda:c.restore(torch.zeros(1,2,3),state,torch.zeros(1,2,2),torch.zeros(1,2,2,2))):
                    with self.assertRaisesRegex(ValueError,'budget'):invoke()

    def test_invalid_geometry_options_and_nonfinite_fail_closed(self):
        for opts in ({'iterations':True},{'iterations':129},{'norm_epsilon':float('nan')},{'connection_epsilon':0},{'max_output_bytes':True}):
            with self.assertRaises(ValueError):component(**opts)
        c=component();state=torch.zeros(1,2,2,3)
        for bad in (state.double(),state[:,:0],state[:,:,:1],state+float('nan'),state+1e30):
            with self.assertRaises(ValueError):c.reduce(bad)
        c.projection[0,0]=float('nan')
        with self.assertRaises(ValueError):c.reduce(state)
        c=component();reduced,post,matrix=c.reduce(state)
        with self.assertRaises(ValueError):c.restore(reduced.double(),state,post,matrix)
        matrix[0,0,0,0]=float('inf')
        with self.assertRaises(ValueError):c.restore(reduced,state,post,matrix)
        with self.assertRaises(ValueError):c.head(state+float('nan'))
