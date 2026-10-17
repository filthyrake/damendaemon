import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_ffn import FlashFeedForward,PackedFlashFeedForward
from damendaemon.glm_ffn import GlmFeedForward
from damendaemon.packed_moe import PackedExpert,PackedMatrix
from tests.test_glm_block import ffn_weights
from tests.test_glm_packed_attention import packed_pair


def control(x,w,sparse,limit):
    def expert(prefix,index=None):
        matrices=[w[prefix+n] for n in ('gate_proj.weight','up_proj.weight','down_proj.weight')]
        if index is not None:matrices=[m[index] for m in matrices]
        g,u,d=matrices;gate=(x@g.T).clamp(max=limit);up=(x@u.T).clamp(-limit,limit)
        return (gate*gate.sigmoid()*up)@d.T
    if not sparse:return expert('')
    probabilities=(x@w['gate.weight'].T).sigmoid()
    ids=sorted(range(len(probabilities)),key=lambda i:(-float(probabilities[i]+w['gate.e_score_correction_bias'][i]),i))[:2]
    denom=sum(float(probabilities[i]) for i in ids)+1e-20
    result=expert('shared_experts.')
    for i in ids:result=result+float(probabilities[i])/denom*2.5*expert('experts.',i)
    return result


def packed_ffn(weights,sparse,limit):
    decoded={}
    def expert(prefix,index=None):
        matrices=[]
        for n in ('gate_proj.weight','up_proj.weight','down_proj.weight'):
            name=prefix+n;value=weights[name] if index is None else weights[name][index]
            p,dense=packed_pair(value)
            if index is None:decoded[name]=dense
            else:
                if name not in decoded:decoded[name]=torch.empty_like(weights[name])
                decoded[name][index]=dense
            matrices.append(PackedMatrix(p.weight,p.scales,p.format))
        return PackedExpert(*matrices,activation_limit=limit,row_chunk=3,column_chunk=2,workspace_bytes=65536)
    if sparse:
        bank=[expert('experts.',i) for i in range(3)];shared=expert('shared_experts.')
        decoded.update({n:weights[n] for n in ('gate.weight','gate.e_score_correction_bias')})
        result=PackedFlashFeedForward(bank,router=decoded['gate.weight'],correction_bias=decoded['gate.e_score_correction_bias'],shared=shared,top_k=2,activation_limit=limit)
    else:result=PackedFlashFeedForward(expert(''),activation_limit=limit)
    return result,decoded


class FlashFfnTests(unittest.TestCase):
    def test_dense_and_routed_shared_clipping_controls(self):
        x=torch.linspace(-20,20,30).reshape(1,5,6);limit=.4
        for sparse in (False,True):
            w=ffn_weights(sparse);c=FlashFeedForward(w,kind='sparse' if sparse else 'dense',top_k=2,activation_limit=limit)
            expected=torch.stack([control(token,w,sparse,limit) for token in x[0]])
            torch.testing.assert_close(c(x)[0],expected,atol=2e-6,rtol=3e-5)
            ordinary=GlmFeedForward(w,kind='sparse' if sparse else 'dense',top_k=2)
            self.assertFalse(torch.allclose(c(x),ordinary(x)))

    def test_packed_clipping_and_post_down_weights(self):
        x=torch.linspace(-12,12,24).reshape(1,4,6)
        for sparse in (False,True):
            packed,w=packed_ffn(ffn_weights(sparse),sparse,.7)
            expected=torch.stack([control(token,w,sparse,.7) for token in x[0]])
            torch.testing.assert_close(packed(x)[0],expected,atol=3e-6,rtol=4e-5)
            for expert in packed.experts:
                with patch.object(expert,'forward',wraps=expert.forward) as forward:
                    packed(x)
                    for call in forward.call_args_list:self.assertNotIn('route_weights',call.kwargs)

    def test_invalid_policies_budget_and_overflow(self):
        w=ffn_weights(False)
        for limit in (None,0,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):FlashFeedForward(w,kind='dense',activation_limit=limit)
        packed,_=packed_ffn(w,False,1.)
        with self.assertRaises(ValueError):PackedFlashFeedForward(packed.experts[0],activation_limit=2.)
        small=FlashFeedForward(w,kind='dense',max_output_bytes=1)
        with self.assertRaises(ValueError):small(torch.ones(1,1,6))
        c=FlashFeedForward(w,kind='dense');c.weights['gate_proj.weight'].fill_(float('inf'))
        with self.assertRaisesRegex(ValueError,'projection'):c(torch.ones(1,1,6))

    def test_clipping_cannot_hide_infinite_packed_projection(self):
        packed,_=packed_ffn(ffn_weights(False),False,1.)
        packed.experts[0].gate.weight.fill_(0x38)
        packed.experts[0].gate.scales.fill_(float('inf'))
        with self.assertRaisesRegex(ValueError,'nonfinite expert projection'):
            packed(torch.ones(1,1,6))
