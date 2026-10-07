import unittest
from unittest.mock import patch
import torch
from dllm.glm_packed_ffn import PackedGlmFeedForward
from dllm.glm_ffn import GlmFeedForward
from dllm.glm_block import GlmBlock
from dllm.glm_model import GlmModel
from dllm.packed_moe import PackedExpert,PackedMatrix
from tests.test_glm_packed_attention import packed_pair,models
from tests.test_glm_block import ffn_weights,dense


def ffns(sparse):
    weights=ffn_weights(sparse);decoded={}
    def expert(prefix,index=None):
        matrices=[]
        for name in ('gate_proj.weight','up_proj.weight','down_proj.weight'):
            key=prefix+name;value=weights[key] if index is None else weights[key][index]
            projection,unpacked=packed_pair(value)
            if index is None:decoded[key]=unpacked
            else:
                if key not in decoded:decoded[key]=torch.empty_like(weights[key])
                decoded[key][index]=unpacked
            matrices.append(PackedMatrix(projection.weight,projection.scales,projection.format))
        return PackedExpert(*matrices,row_chunk=3,column_chunk=2,workspace_bytes=65536)
    if sparse:
        bank=[expert('experts.',index) for index in range(3)];shared=expert('shared_experts.')
        decoded|={name:weights[name] for name in ('gate.weight','gate.e_score_correction_bias')}
        packed=PackedGlmFeedForward(bank,router=decoded['gate.weight'],correction_bias=decoded['gate.e_score_correction_bias'],shared=shared,top_k=2)
    else:packed=PackedGlmFeedForward(expert(''))
    return packed,GlmFeedForward(decoded,kind='sparse' if sparse else 'dense',top_k=2)


def model_pair():
    packed=[];floating=[]
    for index,shared in enumerate((False,True,False)):
        a,b,_=models(shared);f,g=ffns(index>0)
        packed.append(GlmBlock(a,f,torch.ones(6),torch.ones(6)))
        floating.append(GlmBlock(b,g,torch.ones(6),torch.ones(6)))
    generator=torch.Generator().manual_seed(27)
    embedding=torch.randn(13,6,generator=generator);head=torch.randn(13,6,generator=generator)
    return (GlmModel(embedding,blocks,torch.ones(6),head) for blocks in (packed,floating))


class GlmPackedModelTests(unittest.TestCase):
    def test_dense_routed_shared_equations_and_post_down_weighting(self):
        hidden=torch.linspace(-1,1,42).reshape(1,7,6)
        for sparse in (False,True):
            packed,floating=ffns(sparse)
            actual=packed(hidden)
            torch.testing.assert_close(actual,floating(hidden),rtol=3e-5,atol=3e-6)
            if sparse:
                w=floating.weights;expected=[]
                for x in hidden[0]:
                    probabilities=(x@w['gate.weight'].T).sigmoid()
                    ids=(probabilities+w['gate.e_score_correction_bias']).argsort(descending=True,stable=True)[:2]
                    coefficients=probabilities[ids]/(probabilities[ids].sum()+1e-20)*2.5
                    value=dense(x,w['shared_experts.gate_proj.weight'],w['shared_experts.up_proj.weight'],w['shared_experts.down_proj.weight'])
                    for coefficient,index in zip(coefficients,ids):
                        value=value+coefficient*dense(x,w['experts.gate_proj.weight'][index],w['experts.up_proj.weight'][index],w['experts.down_proj.weight'][index])
                    expected.append(value)
                torch.testing.assert_close(actual[0],torch.stack(expected),rtol=3e-5,atol=3e-6)
                with patch.object(packed.experts[0],'forward',wraps=packed.experts[0].forward) as forward:
                    packed(hidden)
                    for call in forward.call_args_list:self.assertNotIn('route_weights',call.kwargs)

    def test_complete_packed_model_incremental_and_reordered_logits(self):
        packed,floating=model_pair();tokens=torch.tensor([[1,3,2,5,4],[2,4,3,6,5]])
        whole,dense_state=floating.prefill(tokens,capacity=5,all_logits=True);self.addCleanup(dense_state.close)
        first,state=packed.prefill(tokens[:,:2],capacity=5,all_logits=True);self.addCleanup(state.close)
        torch.testing.assert_close(first,whole[:,:2],rtol=4e-5,atol=4e-6)
        order=torch.tensor([1,0,1]);state.reorder(order);tokens=tokens[order];whole=whole[order]
        for position in range(2,5):
            actual=packed.forward(tokens[:,position:position+1],state,position=position)
            torch.testing.assert_close(actual,whole[:,position],rtol=4e-5,atol=4e-6)
        self.assertEqual(state.position,5)

    def test_workspace_output_preflight_and_nonfinite_model_failure(self):
        ffn,_=ffns(False);expert=ffn.experts[0]
        with self.assertRaises(ValueError):PackedGlmFeedForward(expert,workspace_bytes=expert.workspace_bytes)
        tiny=PackedGlmFeedForward(expert,max_output_bytes=1)
        with patch.object(expert,'forward',side_effect=AssertionError('projection started')):
            with self.assertRaises(ValueError):tiny(torch.ones(1,1,6))
        packed,_=model_pair();state=packed.new_state(1,2);self.addCleanup(state.close)
        packed.blocks[1].feed_forward.shared.down.scales.fill_(float('nan'))
        with self.assertRaises(ValueError):packed.forward(torch.tensor([[1]]),state,position=0)
        self.assertTrue(state.poisoned);self.assertEqual(state.position,0)
        self.assertTrue(all(layer.poisoned and layer.attention.poisoned for layer in state.layers))

    def test_expert_policy_and_router_validation(self):
        ffn,_=ffns(True)
        for changes in ({'top_k':True},{'route_scale':float('inf')},{'router':ffn.router.to(torch.bfloat16)},{'shared':None}):
            options=dict(router=ffn.router,correction_bias=ffn.correction_bias,shared=ffn.shared,top_k=2)|changes
            with self.assertRaises(ValueError):PackedGlmFeedForward(ffn.experts,**options)
        expert=ffn.experts[0];expert.activation_limit=1.
        with self.assertRaises(ValueError):PackedGlmFeedForward(expert)
