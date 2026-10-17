import unittest

import torch

from damendaemon.floating_attention import attention_prefill, attention_decode
from damendaemon.floating_block import FloatingBlock
from damendaemon.floating_model import FloatingModel
from damendaemon.prepared_projection import PackedProjection
from tests.test_floating_attention import fixture as attention_fixture
from tests.test_packed_projection import fixture as projection_fixture
from tests.test_packed_model import fixture as model_fixture

NAMES = ('wq_a.weight','wq_b.weight','wkv.weight','wo_a.weight','wo_b.weight',
         'indexer.wq_b.weight','indexer.weights_proj.weight')


def prepare(weights,dtype=torch.float32):
    packed,dense = dict(weights),dict(weights)
    for name in NAMES:
        if name not in weights:continue
        rows,columns = weights[name].shape
        _,raw,scales,values = projection_fixture('e4m3fn',rows,columns)
        scales = scales-9
        packed[name] = PackedProjection(raw,scales,format='e4m3fn',dtype=dtype,row_chunk=17,column_chunk=31)
        dense[name] = (values/512).to(dtype)
    return packed,dense


class PackedAttentionTests(unittest.TestCase):
    def test_row_ranges_preserve_original_scale_coordinates_and_strides(self):
        hidden,raw,scales,dense = projection_fixture('e4m3fn',133,160)
        projection=PackedProjection(raw,scales,format='e4m3fn',row_chunk=3,column_chunk=31)
        selected=projection.rows(125,133).rows(1,6)
        source=hidden.unsqueeze(0).expand(2,-1,-1)[:,::2]
        actual=selected.linear(source)
        torch.testing.assert_close(actual,source@dense[126:131].T,rtol=2e-5,atol=1e-3)
        self.assertIs(selected.weight,raw)
        self.assertIs(selected.scales,scales)
        self.assertEqual(selected.linear(source[:0]).shape,(0,4,5))
        for start,stop in ((-1,2),(1,1),(0,134),(True,2)):
            with self.assertRaises(ValueError):projection.rows(start,stop)
        with self.assertRaises(ValueError):projection.linear(hidden.double())

    def test_attention_variants_and_cached_decode_match_dense_control(self):
        for ratio,length in ((0,7),(4,7),(128,129)):
            hidden,weights,freq=attention_fixture(ratio,length)
            packed,dense=prepare(weights)
            options=dict(groups=2,window=4,ratio=ratio,index_top_k=2)
            start=length-2
            actual,state=attention_prefill(hidden[:,:start],packed,freq,max_tokens=length,**options)
            expected,_=attention_prefill(hidden[:,:start],dense,freq,**options)
            torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
            for position in range(start,length):
                actual=attention_decode(hidden[:,position:position+1],state,position=position)
                expected,_=attention_prefill(hidden[:,:position+1],dense,freq,**options)
                torch.testing.assert_close(actual,expected[:,-1:],rtol=4e-5,atol=4e-6)
            state.close()

    def test_packed_attention_and_experts_in_model_stack(self):
        model,_,ids=model_fixture()
        packed_blocks,dense_blocks=[],[]
        for block in model.blocks:
            packed,dense=prepare(block.attention_weights)
            for target,attention in ((packed_blocks,packed),(dense_blocks,dense)):
                target.append(FloatingBlock(attention,block.weights,block.frequencies,groups=2,window=4,
                                             ratio=block.ratio,hc_mult=2,top_k=2,index_top_k=2,activation_limit=2,
                                             packed_experts=block.packed_experts,packed_shared=block.packed_shared))
        packed=FloatingModel(model.embedding,packed_blocks,model.head)
        control=FloatingModel(model.embedding,dense_blocks,model.head)
        actual,state=packed.prefill(ids[:,:3],max_tokens=5)
        expected,_=control.prefill(ids[:,:3])
        torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        for position in (3,4):
            actual=packed.decode(ids[:,position:position+1],state,position=position)
            expected,_=control.prefill(ids[:,:position+1])
            torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        state.close()

    def test_low_precision_adapter_and_strict_norm_contract(self):
        hidden,weights,freq=attention_fixture(0,4,torch.bfloat16)
        packed,_=prepare(weights,torch.bfloat16)
        actual,state=attention_prefill(hidden,packed,freq,groups=2,window=4)
        self.assertEqual(actual.dtype,torch.bfloat16)
        state.close()
        packed['wq_a.weight']=PackedProjection(packed['wq_a.weight'].weight,packed['wq_a.weight'].scales,format='e4m3fn')
        with self.assertRaises(ValueError):attention_prefill(hidden,packed,freq,groups=2,window=4)
        packed,_=prepare(weights,torch.bfloat16)
        packed['q_norm.weight']=packed['wq_a.weight']
        with self.assertRaises(ValueError):attention_prefill(hidden,packed,freq,groups=2,window=4)
