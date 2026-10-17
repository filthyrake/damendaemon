import math
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_kda import recurrent_attention


def inputs():
    torch.manual_seed(771)
    return (torch.randn(2,5,2,3),torch.randn(2,5,2,3),torch.randn(2,5,2,4),
            -torch.rand(2,5,2,3)*5,torch.rand(2,5,2),torch.randn(2,2,3,4)*.2)


def scalar_control(q,key,value,g,beta,initial):
    b,t,h,k=q.shape;v=value.shape[-1];state=initial.double().clone();out=torch.empty(b,t,h,v,dtype=torch.float64)
    for batch in range(b):
        for token in range(t):
            for head in range(h):
                qq=q[batch,token,head].tolist();kk=key[batch,token,head].tolist()
                qq=[x/math.sqrt(sum(y*y for y in qq)+1e-6)/math.sqrt(k) for x in qq]
                kk=[x/math.sqrt(sum(y*y for y in kk)+1e-6) for x in kk]
                for i in range(k):
                    for j in range(v):state[batch,head,i,j]*=math.exp(float(g[batch,token,head,i]))
                delta=[(float(value[batch,token,head,j])-sum(float(state[batch,head,i,j])*kk[i] for i in range(k)))*float(beta[batch,token,head]) for j in range(v)]
                for i in range(k):
                    for j in range(v):state[batch,head,i,j]+=kk[i]*delta[j]
                for j in range(v):out[batch,token,head,j]=sum(qq[i]*float(state[batch,head,i,j]) for i in range(k))
    return out.float(),state.float()


class FlashKdaTests(unittest.TestCase):
    def test_independent_scalar_recurrence_and_no_mutation(self):
        args=inputs();copies=[x.clone() for x in args]
        actual=recurrent_attention(*args);expected=scalar_control(*args)
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=2e-7,rtol=3e-6)
        for a,e in zip(args,copies):torch.testing.assert_close(a,e)
        self.assertNotEqual(actual[1].data_ptr(),args[-1].data_ptr())

    def test_full_chunk_and_single_token_continuation(self):
        args=inputs();expected,state=recurrent_attention(*args)
        for chunks in ((2,3),(1,1,1,1,1)):
            cursor=0;current=args[-1];out=[]
            for length in chunks:
                y,current=recurrent_attention(*(x[:,cursor:cursor+length] for x in args[:-1]),current)
                out.append(y);cursor+=length
            torch.testing.assert_close(torch.cat(out,1),expected,rtol=0,atol=0)
            torch.testing.assert_close(current,state,rtol=0,atol=0)
        zeros=torch.zeros_like(args[-1])
        for a,b in zip(recurrent_attention(*args[:-1]),recurrent_attention(*args[:-1],zeros)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_dimension_specific_decay_with_zero_update(self):
        q=torch.ones(1,1,1,2);key=q.clone();v=torch.zeros(1,1,1,2)
        g=torch.tensor([[[[0.,-math.log(2)]]]]);beta=torch.zeros(1,1,1);initial=torch.tensor([[[[1.,2.],[3.,4.]]]])
        out,state=recurrent_attention(q,key,v,g,beta,initial)
        torch.testing.assert_close(state,torch.tensor([[[[1.,2.],[1.5,2.]]]]))
        torch.testing.assert_close(out,torch.tensor([[[[2.5,4.]]]])/math.sqrt(2+1e-6)/math.sqrt(2))

    def test_preallocation_budget_and_invalid_inputs(self):
        args=inputs()
        for option in ('max_workspace_bytes','max_output_bytes'):
            with patch('damendaemon.glm_flash_kda._finite',side_effect=AssertionError('scan before admission')):
                with self.assertRaisesRegex(ValueError,'budget'):recurrent_attention(*args,**{option:1})
        for index,bad in ((0,args[0].double()),(1,args[1][:,:,:,:2]),(3,args[3]+10),(4,args[4]+2),
                          (5,args[5][:,:,:,:2]),(0,args[0]*1e30),(2,args[2]+float('nan'))):
            changed=list(args);changed[index]=bad
            with self.assertRaises(ValueError):recurrent_attention(*changed)
        initial=args[-1].clone();bad=list(args);bad[2]=args[2]+float('nan')
        with self.assertRaises(ValueError):recurrent_attention(*bad)
        torch.testing.assert_close(initial,args[-1])
