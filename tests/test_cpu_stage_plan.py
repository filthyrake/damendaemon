import itertools
from fractions import Fraction
import unittest
from unittest.mock import patch
import torch
from damendaemon.cpu_hybrid_stage import cpu_stage_budget,CpuHybridStage
from damendaemon.cpu_stage_plan import plan_cpu_stages
from tests.test_cpu_hybrid_stage import model

OPTIONS=dict(max_batch=2,projection_workspace_bytes=2**20,max_workspace_bytes=2**20,max_output_bytes=65536)


class CpuStagePlanTests(unittest.TestCase):
    def test_packed_scale_and_alias_accounting_without_allocation(self):
        source=model('q4_0');source.weights['2.qkv']=source.weights['0.qkv'];source.scales['2.qkv']=source.scales['0.qkv']
        with patch('torch.zeros',side_effect=AssertionError('allocated')),patch('torch.empty',side_effect=AssertionError('allocated')):
            result=cpu_stage_budget(source,0,4,**OPTIONS)
        expected=sum(t.numel()*t.element_size() for t in {id(t):t for t in source.weights.values()}.values())
        expected+=sum(t.numel()*t.element_size() for t in {id(t):t for t in source.scales.values()}.values())
        self.assertEqual(result['weight_bytes'],expected)
        self.assertEqual(result['required_memory_bytes'],sum(result[k] for k in ('weight_bytes','conversion_overlap_bytes','state_bytes','rotary_bytes','workspace_bytes','output_bytes')))

    def test_float_promotion_preserves_aliases_and_peak_accounting(self):
        source=model();source.weights={k:v.half() for k,v in source.weights.items()}
        source.weights['2.qkv']=source.weights['0.qkv']
        budget=cpu_stage_budget(source,0,4,**OPTIONS)
        self.assertEqual(budget['weight_bytes'],sum(v.numel()*4 for v in {id(v):v for v in source.weights.values()}.values()))
        self.assertEqual(budget['conversion_overlap_bytes'],max(v.numel()*2 for v in source.weights.values()))
        stage=CpuHybridStage(source,0,4,**OPTIONS);self.addCleanup(stage.close)
        self.assertIs(stage.w['0.qkv'],stage.w['2.qkv']);self.assertEqual(stage.required_memory_bytes,budget['required_memory_bytes'])

    def test_placement_matches_exhaustive_partitions_and_constructs(self):
        for mode in ('none','q4_0','q8_0'):
            source=model(mode)
            for budgets in ((2**22,), (2**21,2**22),(2**22,2**21,2**22)):
                possibilities=[]
                for splits in itertools.combinations(range(1,4),len(budgets)-1):
                    edges=(0,*splits,4);ranges=tuple(zip(edges,edges[1:]))
                    costs=[cpu_stage_budget(source,a,b,**OPTIONS)['required_memory_bytes'] for a,b in ranges]
                    if all(cost<=limit for cost,limit in zip(costs,budgets)):
                        possibilities.append((max(Fraction(cost,limit) for cost,limit in zip(costs,budgets)),ranges))
                result=plan_cpu_stages(source,budgets,**OPTIONS)
                self.assertEqual(result['storage_plan_found'],bool(possibilities))
                if not possibilities:continue
                score,ranges=min(possibilities)
                self.assertEqual(result['splits'],[b for _,b in ranges[:-1]])
                self.assertEqual(result['maximum_estimated_utilization'],float(score))
                for item in result['stages']:
                    stage=CpuHybridStage(source,item['layer_start'],item['layer_end'],**OPTIONS)
                    self.assertEqual(stage.required_memory_bytes,item['required_memory_bytes']);stage.close()

    def test_partial_promotion_cleanup_preserves_source(self):
        source=model();stage=CpuHybridStage.__new__(CpuHybridStage)
        original=dict(source.weights)
        with patch.object(torch.Tensor,'float',side_effect=MemoryError('promotion failed')):
            with self.assertRaises(MemoryError):stage.__init__(source,0,4,**OPTIONS)
        stage.release_partial()
        self.assertEqual(set(source.weights),set(original))
        for name,value in original.items():self.assertIs(source.weights[name],value)

    def test_no_fit_does_not_hide_invalid_storage_or_options(self):
        source=model('q8_0')
        self.assertFalse(plan_cpu_stages(source,[1,1],**OPTIONS)['storage_plan_found'])
        self.assertFalse(plan_cpu_stages(source,[2**30,2**30],max_weight_bytes=1,**OPTIONS)['storage_plan_found'])
        small=dict(OPTIONS,projection_workspace_bytes=65536)
        self.assertFalse(plan_cpu_stages(source,[2**30,2**30],**small)['storage_plan_found'])
        for budgets in ([True],[],[0],[1]*5):
            with self.assertRaises(ValueError):plan_cpu_stages(source,budgets,**OPTIONS)
        with self.assertRaises(ValueError):plan_cpu_stages(source,[1],max_batch=True)
        source.scales['3.qkv']=source.scales['3.qkv'].float()
        with self.assertRaises(ValueError):plan_cpu_stages(source,[1,1],**OPTIONS)
