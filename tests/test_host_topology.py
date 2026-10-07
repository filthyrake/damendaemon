from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest
from dllm.host_topology import HostTopology
from tests.test_cpu_hybrid_stage import model


class HostTopologyTests(unittest.TestCase):
    def setUp(self):self.config=model().config

    def plan(self,**options):
        return HostTopology(self.config,[(0,1),(1,2),(2,4)],b'm'*32,context=16,chunk_size=8,frame_bytes=4096,**options)

    def test_stable_identity_policy_and_immutable_input_snapshot(self):
        plan=self.plan(mapped_vocab_size=60,excluded_ids=[59])
        expected=dict(version=1,backend='cpu-float32',config=self.config,ranges=[(0,1),(1,2),(2,4)],
            source_identity=(b'm'*32).hex(),quantization='none',context=16,context_limit=512,slots=4,
            chunk_size=8,frame_bytes=4096,mapped_vocab_size=60,excluded_ids=[59])
        identity=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
        self.assertEqual(plan.identity,identity)
        self.assertEqual(plan.identity,self.plan(mapped_vocab_size=60,excluded_ids={59}).identity)
        self.assertEqual([plan.result_kind(i) for i in range(3)],['activation','activation','tokens'])
        self.assertEqual(len({plan.stage_identity(i) for i in range(3)}),3)
        self.assertNotEqual(plan.identity,self.plan(mapped_vocab_size=60,excluded_ids=[58]).identity)
        self.assertNotEqual(plan.identity,self.plan(mapped_vocab_size=60,excluded_ids=[59],quantization='q4_0').identity)
        self.config['layer_types'][0]='full_attention'
        self.assertEqual(plan.config['layer_types'][0],'linear_attention')
        changed=plan.config;changed['layer_types'].clear();self.assertEqual(len(plan.config['layer_types']),4)
        with self.assertRaises(FrozenInstanceError):plan.context=32

    def test_rejects_incomplete_overlapping_or_unbounded_topologies(self):
        for ranges in ([(0,4)],[(0,1),(2,4)],[(0,2),(1,4)],[(0,2),(2,3)],[(1,2),(2,4)],
                       [(0,0),(0,4)],[(False,1),(1,4)],[(0,1)]*9):
            with self.subTest(ranges=ranges),self.assertRaises(ValueError):HostTopology(self.config,ranges,b'm'*32,context=16)
        for options in ({'context':65},{'slots':5},{'chunk_size':0},{'frame_bytes':255},
                        {'frame_bytes':256,'chunk_size':8},{'quantization':'fp4'},{'mapped_vocab_size':65},
                        {'mapped_vocab_size':1,'excluded_ids':[0]},{'context_limit':15}):
            args=dict(context=16,chunk_size=8,frame_bytes=4096)|options
            with self.subTest(options=options),self.assertRaises(ValueError):HostTopology(self.config,[(0,2),(2,4)],b'm'*32,**args)
        for index in (-1,3,True):
            with self.assertRaises(ValueError):self.plan().stage_identity(index)
