import unittest
import torch
from damendaemon.glm_cached_attention import CachedGlmAttention
from damendaemon.prepared_projection import PackedProjection
from tests.test_glm_cached_attention import fixture
from tests.test_packed_fp32_scales import scalar


def packed_pair(value):
    raw=(value/.125).to(torch.float8_e4m3fn).view(torch.uint8)
    scales=torch.full(((value.shape[0]+127)//128,(value.shape[1]+127)//128),.125)
    dense=torch.tensor([[scalar(int(code))*.125 for code in row] for row in raw])
    return PackedProjection(raw,scales,format='e4m3fn_f32',row_chunk=3,column_chunk=2),dense


def models(shared=False):
    original,hidden=fixture(shared)
    packed={};dense={}
    for name,value in original.weights.items():
        if value.ndim==2 and name!='indexer.weights_proj.weight':
            packed[name],dense[name]=packed_pair(value)
        else:packed[name]=dense[name]=value
    options=dict(heads=2,nope_dim=2,rope_dim=4,value_dim=3,index_top_k=3,
                 indexer_type='shared' if shared else 'full')
    return CachedGlmAttention(packed,**options),CachedGlmAttention(dense,**options),hidden


class GlmPackedAttentionTests(unittest.TestCase):
    def test_full_shared_chunked_cache_reorder_matches_dense_control(self):
        packed,dense,hidden=models();shared,shared_dense,_=models(True)
        states=[m.new_state(2,7) for m in (packed,dense,shared,shared_dense)]
        for state in states:self.addCleanup(state.close)
        original=packed.weights['q_b_proj.weight'].weight.clone()
        for start,stop in ((0,2),(2,5),(5,7)):
            x=hidden[:,start:stop]
            a,ids=packed.forward(x,states[0],position=start)
            b,expected_ids=dense.forward(x,states[1],position=start)
            torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(ids,expected_ids,rtol=0,atol=0)
            c,reused=shared.forward(x,states[2],position=start,previous_indices=ids)
            d,_=shared_dense.forward(x,states[3],position=start,previous_indices=expected_ids)
            torch.testing.assert_close(c,d,rtol=3e-5,atol=3e-6);self.assertIs(reused,ids)
            fresh,_,cache=dense.prefill(hidden[:,:stop],capacity=7)
            try:torch.testing.assert_close(a,fresh[:,start:stop],rtol=3e-5,atol=3e-6)
            finally:cache.close()
            if stop==2:
                order=torch.tensor([1,0,1]);hidden=hidden[order]
                for state in states:state.reorder(order)
        torch.testing.assert_close(packed.weights['q_b_proj.weight'].weight,original,rtol=0,atol=0)

    def test_nonfinite_packed_failure_before_and_after_cache_mutation(self):
        for name,poisoned in (('q_a_proj.weight',False),('o_proj.weight',True),('indexer.wk.weight',False),('indexer.wq_b.weight',True)):
            model,_,hidden=models();state=model.new_state(2,7);self.addCleanup(state.close)
            model.weights[name].scales.fill_(float('nan'))
            with self.assertRaises(ValueError):model.forward(hidden[:,:1],state,position=0)
            self.assertEqual(state.poisoned,poisoned);self.assertEqual(state.position,0)

    def test_reject_implicit_scale_format_and_non_fp32_adapter(self):
        model,_,_=models()
        for format,dtype in (('e4m3fn',torch.float32),('e4m3fn_f32',torch.bfloat16)):
            weights=model.weights.copy();old=weights['q_a_proj.weight']
            scales=old.scales if format=='e4m3fn_f32' else torch.full(old.scales.shape,127,dtype=torch.uint8)
            weights['q_a_proj.weight']=PackedProjection(old.weight,scales,format=format,dtype=dtype)
            with self.assertRaises(ValueError):CachedGlmAttention(weights,heads=2,nope_dim=2,rope_dim=4,value_dim=3,index_top_k=3)
