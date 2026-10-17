"""Packed stage execution against independently expanded fixture controls."""
import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from damendaemon.cpu_hybrid_stage import CpuHybridStage
from damendaemon.quantization import quantize,quantize_q4,dequantize,dequantize_q4
from damendaemon.host_hybrid_worker import HostHybridWorker
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.stage_protocol import StageState
from tests import test_host_tensor_channel


def model(mode='none'):
    torch.manual_seed(104)
    c=dict(model_type='qwen3_5_text',hidden_size=32,vocab_size=64,num_hidden_layers=4,
           num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=1,
           linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
           linear_conv_kernel_dim=4,intermediate_size=64,rotary_dim=4,max_position_embeddings=64,
           layer_types=['linear_attention','full_attention']*2,rms_norm_eps=1e-6,rope_theta=10000.)
    w={'embed':torch.randn(64,32)*.1,'head':torch.randn(64,32)*.1,'norm':torch.ones(32)}
    for i,kind in enumerate(c['layer_types']):
        shapes={'norm1':(32,),'norm2':(32,),'gate_up':(128,32),'down':(32,64)}
        shapes.update({'qkv':(96,32),'o':(32,32),'q_norm':(8,),'k_norm':(8,)} if kind=='full_attention' else
                      {'qkv':(64,32),'z':(32,32),'b':(2,32),'a':(2,32),'out':(32,32),'conv':(64,1,4),
                       'dt':(2,),'A':(2,),'gated_norm':(16,)})
        for name,shape in shapes.items():w[f'{i}.{name}']=torch.ones(shape) if 'norm' in name else torch.randn(shape)*.1
    scales={}
    if mode!='none':
        for name,value in list(w.items()):
            if value.ndim==2 and name!='embed':w[name],scales[name]=(quantize_q4 if mode=='q4_0' else quantize)(value)
    return SimpleNamespace(config=c,weights=w,scales=scales,quantization=mode,max_context=16,vision_weights=None)


def decoded(source):
    result=copy.copy(source);result.weights=dict(source.weights);result.scales={};result.quantization='none'
    for name,scale in source.scales.items():
        result.weights[name]=(dequantize_q4 if source.quantization=='q4_0' else dequantize)(source.weights[name],scale)
    return result


class CpuHybridStageTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def stage(self,source,start=0,end=4,**options):
        result=CpuHybridStage(source,start,end,max_batch=2,**options);self.addCleanup(result.close);return result

    def compare_state(self,stages,full):
        for stage in stages:
            for name in ('kv','conv','state'):
                for i,value in getattr(stage,name).items():torch.testing.assert_close(value,getattr(full,name)[i],rtol=2e-4,atol=2e-6)

    def test_split_chunks_decode_and_packed_ownership(self):
        for mode in ('q4_0','q8_0'):
            source=model(mode);left=self.stage(source,0,2);right=self.stage(source,2,4);full=self.stage(decoded(source))
            for ids,positions,slots,prefill,prefix in (([[1,2],[3,4]],[[0,1],[0,1]],[0,1],True,0),
                    ([[5,6],[7,8]],[[2,3],[2,3]],[0,1],True,2),([[9],[10]],[[4],[4]],[1,0],False,0)):
                ids=torch.tensor(ids);positions=torch.tensor(positions);slots=torch.tensor(slots)
                expected=full.forward_layers(full.embedding(ids),positions,slots,prefill,prefix)
                actual=right.forward_layers(left.forward_layers(left.embedding(ids),positions,slots,prefill,prefix),positions,slots,prefill,prefix)
                torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
                torch.testing.assert_close(right.output_logits(actual[:,-1]),full.output_logits(expected[:,-1]),rtol=2e-4,atol=2e-6)
                self.compare_state((left,right),full)
            for stage in (left,right):
                for name,scale in stage.model.scales.items():
                    self.assertIs(stage.w[name],source.weights[name]);self.assertIs(scale,source.scales[name])
            self.assertNotIn('head',left.w);self.assertNotIn('embed',right.w)
            left.close();left.close();self.assertFalse(left.adapters);self.assertFalse(left.model.scales);self.assertTrue(source.scales)

    def test_budgets_and_bad_storage_reject_before_cache_allocation(self):
        for options in (dict(max_weight_bytes=1),dict(max_cache_bytes=1),dict(max_workspace_bytes=1)):
            with patch('torch.zeros',side_effect=AssertionError('allocated')):
                with self.assertRaisesRegex(ValueError,'budget'):CpuHybridStage(model('q4_0'),0,2,**options)
        source=model('q4_0');source.scales['0.qkv']=source.scales['0.qkv'].float()
        with self.assertRaises(ValueError):self.stage(source,0,2)
        source=model();source.weights['0.conv']=torch.zeros(64,4)
        with self.assertRaises(ValueError):self.stage(source,0,2)

    def test_failure_poison_and_invalid_geometry(self):
        stage=self.stage(model('q8_0'))
        with self.assertRaises(ValueError):stage.embedding(torch.tensor([1]))
        self.assertFalse(stage.failed)
        with patch.object(stage,'attention',side_effect=RuntimeError('after recurrent write')):
            with self.assertRaisesRegex(RuntimeError,'recurrent'):stage.forward_layers(stage.embedding(torch.tensor([[1,2]])),torch.tensor([[0,1]]),torch.tensor([0]),True)
        self.assertTrue(stage.failed)
        with self.assertRaises(RuntimeError):stage.embedding(torch.tensor([[1]]))

    def test_normalization_overflow_and_head_preflight(self):
        stage=self.stage(model())
        with self.assertRaisesRegex(ValueError,'variance'):
            stage.forward_layers(torch.full((1,1,32),1e30),torch.tensor([[0]]),torch.tensor([0]),True)
        self.assertTrue(stage.failed)
        stage=self.stage(model(),max_output_bytes=1)
        with patch.object(stage,'norm',side_effect=AssertionError('normalized before admission')):
            with self.assertRaisesRegex(ValueError,'geometry/budget'):stage.output_logits(torch.zeros(1,32))

    def parallel(self,functions):
        with ThreadPoolExecutor(2) as pool:return [f.result(timeout=10) for f in [pool.submit(call) for call in functions]]

    def workers(self,source,other=None):
        transports=test_host_tensor_channel.HostTensorChannelTests.pair(self,caps=(65536,65536),timeout=2)
        stages=[self.stage(source,0,2),self.stage(other or source,2,4)]
        workers=[HostHybridWorker(HostStageChannel(t,StageState(1,16,2),vocab_size=64),s) for t,s in zip(transports,stages)]
        for worker in workers:self.addCleanup(worker.close)
        return workers

    def test_loopback_prefill_decode_release_and_format_agreement(self):
        for mode in ('q4_0','q8_0'):
            source=model(mode);left,right=self.workers(source);self.parallel([left.handshake,right.handshake]);full=self.stage(decoded(source))
            for tokens,start in (([1,2],0),([3,4],2)):
                positions=torch.arange(start,start+2).reshape(1,2)
                hidden=full.forward_layers(full.embedding(torch.tensor([tokens])),positions,torch.tensor([0]),True,start)
                expected=int(full.output_logits(hidden[:,-1]).argmax(-1)[0])
                self.assertEqual(self.parallel([lambda:left.prefill_chunk(tokens,0,start),right.step])[0],expected)
            hidden=full.forward_layers(full.embedding(torch.tensor([[5]])),torch.tensor([[4]]),torch.tensor([0]),False)
            self.assertEqual(self.parallel([lambda:left.decode([5],[0]),right.step])[0],full.output_logits(hidden[:,-1]).argmax(-1).tolist())
            self.compare_state((left.stage,right.stage),full)
            self.parallel([lambda:left.release([0]),right.step])
            for stage in (left.stage,right.stage):
                for group in (stage.conv,stage.state):
                    for value in group.values():self.assertEqual(value[0].count_nonzero().item(),0)
            self.parallel([left.stop,right.step])
        left,right=self.workers(model('q4_0'),model('q8_0'))
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(w.handshake) for w in (left,right)]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,OSError,EOFError)):future.result(timeout=10)
