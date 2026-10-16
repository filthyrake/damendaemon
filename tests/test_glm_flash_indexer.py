import math
import unittest
from unittest.mock import patch
import torch
from dllm.glm_flash_indexer import FlashIndexer


def indexer(**options):
    torch.manual_seed(905)
    weights={'wk.weight':torch.randn(3,4),'wq_b.weight':torch.randn(6,2),
             'weights_proj.weight':torch.randn(2,4),'k_norm.weight':torch.randn(3),'k_norm.bias':torch.randn(3),
             'index_kpool_compress_ape':torch.randn(2,3),'index_kpool_compress_gate':torch.randn(3,4)}
    return FlashIndexer(weights,heads=2,pool_size=2,top_k=2,**options)


class FlashIndexerTests(unittest.TestCase):
    def test_visible_dot_overflow_rejected_before_relu(self):
        c=indexer();c.weights['wq_b.weight'].fill_(1)
        hidden=torch.ones(1,1,4);low=torch.full((1,1,2),1e20)
        gates=torch.zeros(1,2,3);positions=torch.tensor([[1]])
        for sign in (-1,1):
            with self.subTest(sign=sign),self.assertRaisesRegex(ValueError,'nonfinite'):
                c.select(hidden,low,torch.full((1,2,3),sign*1e20),gates,positions)

    def test_future_pool_dots_are_not_evaluated(self):
        c=indexer();c.weights['wq_b.weight'].fill_(1)
        hidden=torch.ones(1,1,4);low=torch.full((1,1,2),1e20)
        keys=torch.ones(1,4,3);keys[:,2:]=1e20
        actual=c.select(hidden,low,keys,torch.zeros_like(keys),torch.tensor([[1]]))
        self.assertEqual(actual.tolist(),[[[0,1,-1]]])

    def test_learned_pool_and_scores_independent_scalar_control(self):
        c=indexer();hidden=torch.randn(1,7,4);low=torch.randn(1,7,2);keys,gates=c.project_keys(hidden)
        raw=hidden@c.weights['wk.weight'].T
        expected=(raw-raw.mean(-1,keepdim=True))/torch.sqrt(raw.var(-1,unbiased=False,keepdim=True)+1e-6)
        torch.testing.assert_close(keys,expected*c.weights['k_norm.weight']+c.weights['k_norm.bias'])
        positions=torch.arange(7).reshape(1,7);actual=c.select(hidden,low,keys,gates,positions)
        for token in range(7):
            scores=[]
            for start in range(0,token,2):
                pooled=[]
                for channel in range(3):
                    logits=[float(gates[0,start+j,channel]+c.weights['index_kpool_compress_ape'][j,channel]) for j in range(2)]
                    probs=[math.exp(v-max(logits)) for v in logits]
                    pooled.append(sum(probs[j]*float(keys[0,start+j,channel]) for j in range(2))/sum(probs))
                score=0
                for head in range(2):
                    query=[sum(float(low[0,token,r])*float(c.weights['wq_b.weight'][head*3+d,r]) for r in range(2)) for d in range(3)]
                    weight=sum(float(hidden[0,token,r])*float(c.weights['weights_proj.weight'][head,r]) for r in range(4))/math.sqrt(2)
                    score+=max(0,sum(a*b for a,b in zip(query,pooled))/math.sqrt(3))*weight
                scores.append(score)
            expected=[-1,-1,token if (token+1)%2 else -1]
            if scores:
                best=max(range(len(scores)),key=scores.__getitem__);expected[:2]=[2*best,2*best+1]
            self.assertEqual(actual[0,token].tolist(),expected)

    def test_left_padding_tail_boundaries_causality_and_ties(self):
        c=indexer();hidden=torch.zeros(2,8,4);low=torch.zeros(2,8,2);keys,gates=c.project_keys(hidden)
        valid=torch.tensor([[False,False,True,True,True,True,True,False],[False]*8])
        positions=torch.arange(8).expand(2,-1)
        actual=c.select(hidden,low,keys,gates,positions,valid_keys=valid)
        self.assertEqual(actual[0].tolist(),[[-1,-1,-1],[-1,-1,-1],[-1,-1,2],[2,3,-1],[2,3,4],[2,3,-1],[2,3,6],[-1,-1,-1]])
        self.assertTrue((actual[1]==-1).all())
        for token in range(8):
            part=c.select(hidden[:,:token+1],low[:,:token+1],keys[:,:token+1],gates[:,:token+1],positions[:,:token+1],valid_keys=valid[:,:token+1])
            torch.testing.assert_close(part,actual[:,:token+1])

    def test_masks_nonfinite_geometry_and_preallocation_limits(self):
        c=indexer();hidden=torch.randn(1,4,4);low=torch.randn(1,4,2);keys,gates=c.project_keys(hidden);pos=torch.arange(4).reshape(1,4)
        with self.assertRaisesRegex(ValueError,'holes'):c.select(hidden,low,keys,gates,pos,valid_keys=torch.tensor([[True,False,True,True]]))
        for args in ((hidden,low,keys,gates,pos+1),(hidden,low.double(),keys,gates,pos),(hidden,low,keys+float('nan'),gates,pos)):
            with self.assertRaises(ValueError):c.select(*args)
        c.weights['weights_proj.weight'][0,0]=float('nan')
        with self.assertRaises(ValueError):c.select(hidden,low,keys,gates,pos)
        for option in ('max_workspace_bytes','max_output_bytes'):
            c=indexer(**{option:1})
            with patch('dllm.glm_flash_indexer._finite',side_effect=AssertionError('scanned before admission')):
                with self.assertRaisesRegex(ValueError,'budget'):c.project_keys(hidden)
                with self.assertRaisesRegex(ValueError,'budget'):c.select(hidden,low,keys,gates,pos)
