from contextlib import ExitStack,redirect_stdout,redirect_stderr
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.reference_stage_plan import inspect_reference_storage,plan_reference_stages
from dllm.safetensor_header import ITEM_BYTES
from tests import test_glm_reference_loader as glm,test_glm_flash_reference_loader as flash,test_deepseek_reference_loader as deepseek
from scripts import plan_reference_host_topology as command


class ReferenceStoragePlanTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def snapshot(self,family):
        path=self.path/family;path.mkdir(exist_ok=True)
        result={'glm':glm,'flash':flash,'deepseek':deepseek}[family].fixture(path)
        return path,result[0],result[1]

    def no_payload(self):
        stack=ExitStack()
        for family in ('glm','glm_flash','deepseek'):
            stack.enter_context(patch('dllm.'+family+'_reference_loader._payload',side_effect=AssertionError('payload')))
        stack.enter_context(patch('torch.cuda.is_available',side_effect=AssertionError('device probe')))
        return stack

    def test_header_weight_and_loaded_cache_reservations_for_every_family(self):
        for family in ('glm','flash','deepseek'):
            path,config,headers=self.snapshot(family)
            with self.no_payload():
                layout=inspect_reference_storage(path,family)
                report=plan_reference_stages(layout,[2**30]*(2 if family=='glm' else 3),context=129,slots=2,chunk_size=128)
            self.assertTrue(report['storage_plan_found']);self.assertFalse(report['execution_admitted']);self.assertFalse(report['source_model_identity_verified'])
            if family=='glm':
                from dllm.glm_reference_loader import load_reference_stage
                from dllm.glm_checkpoint import tensor_specs
                prefix='model.layers.';embedding='model.embed_tokens.weight'
            elif family=='flash':
                from dllm.glm_flash_reference_loader import load_reference_stage
                from dllm.glm_flash_checkpoint import tensor_specs,ROOT
                prefix=ROOT+'layers.';embedding=ROOT+'embed_tokens.weight'
            else:
                from dllm.deepseek_reference_loader import load_reference_stage
                from dllm.deepseek_checkpoint import tensor_specs
                prefix='layers.';embedding='embed.weight'
            specs=tensor_specs(config)
            for item in report['stages']:
                start,end=item['layers'];stage=load_reference_stage(path,start,end)
                selected={n:s for n,s in specs.items() if (n==embedding and start==0) or (not n.startswith(prefix) and n!=embedding and end==3) or n.startswith(tuple(prefix+str(i)+'.' for i in range(start,end)))}
                retained=sum(math.prod(s.shape)*(4 if s.storage=='floating' or family!='deepseek' and s.storage=='scale' else ITEM_BYTES[headers[n]['dtype']] if s.storage=='integer' else 1) for n,s in selected.items())
                self.assertEqual(item['retained_weight_bytes'],retained)
                self.assertEqual(item['source_conversion_overlap_bytes'],max(math.prod(s.shape)*ITEM_BYTES[headers[n]['dtype']] for n,s in selected.items()))
                if family=='deepseek':
                    hidden=None if stage.first else torch.zeros(2,1,stage.streams,stage.dim)
                    _,state=stage.prefill(torch.ones(2,1,dtype=torch.int64),hidden=hidden,max_tokens=129)
                    actual=2*sum(s.attention.bytes_per_slot for s in state.layers)
                else:
                    state=stage.new_state(2,129)
                    if family=='glm':actual=2*sum(s.attention.bytes_per_slot for s in state.layers)
                    else:actual=2*sum(b.attention.bytes_per_slot if hasattr(b.attention,'bytes_per_slot') else 129*b.attention.bytes_per_token for b in stage.blocks)
                self.assertEqual(item['cache_reservation_bytes'],actual);state.close();stage.cleanup()
                total=sum(item[k] for k in ('retained_weight_bytes','source_conversion_overlap_bytes','cache_reservation_bytes','projection_workspace_bytes','component_workspace_allowance_bytes','output_allowance_bytes','transport_buffer_allowance_bytes'))
                self.assertEqual(item['required_storage_and_allowance_bytes'],total)

    def test_optimal_ordered_partitions_match_exhaustive_control_and_exact_no_fit(self):
        for family in ('flash','deepseek'):
            path,_,_=self.snapshot(family);layout=inspect_reference_storage(path,family)
            single=plan_reference_stages(layout,[2**30]*3,context=128,slots=2)['stages']
            def cost(start,end):
                selected=single[start:end]
                return (sum(s['retained_weight_bytes']+s['cache_reservation_bytes'] for s in selected)
                    +max(s['source_conversion_overlap_bytes'] for s in selected)
                    +sum(selected[0][k] for k in ('projection_workspace_bytes','component_workspace_allowance_bytes','output_allowance_bytes','transport_buffer_allowance_bytes')))
            base=max(cost(0,1),cost(1,3),cost(0,2),cost(2,3))
            for budgets in ([base,base],[base,base+10000],[base+10000,base],[1,base]):
                choices=[]
                for cut in (1,2):
                    ranges=((0,cut),(cut,3));values=[cost(*r) for r in ranges]
                    if all(v<=limit for v,limit in zip(values,budgets)):choices.append((max(Fraction(v,limit) for v,limit in zip(values,budgets)),ranges))
                actual=plan_reference_stages(layout,budgets,context=128,slots=2)
                self.assertEqual(actual['storage_plan_found'],bool(choices))
                if choices:self.assertEqual(actual['plan']['ranges'],[list(r) for r in min(choices)[1]])
            exact=[s['required_storage_and_allowance_bytes'] for s in single]
            self.assertTrue(plan_reference_stages(layout,exact,context=128,slots=2)['storage_plan_found'])
            self.assertFalse(plan_reference_stages(layout,[exact[0]-1,*exact[1:]],context=128,slots=2)['storage_plan_found'])
        path,_,_=self.snapshot('glm');layout=inspect_reference_storage(path,'glm')
        self.assertEqual(plan_reference_stages(layout,[2**30]*2)['plan']['ranges'],[[0,2],[2,3]])
        self.assertFalse(plan_reference_stages(layout,[2**30]*3)['storage_plan_found'])

    def test_resource_no_fit_preserves_context_slots_and_bad_options_are_errors(self):
        path,_,_=self.snapshot('deepseek');layout=inspect_reference_storage(path,'deepseek')
        for options in (dict(max_weight_bytes=1),dict(max_cache_bytes=1),dict(max_output_bytes=1),dict(frame_bytes=256)):
            with self.no_payload():result=plan_reference_stages(layout,[2**30]*2,context=32768,slots=4,**options)
            self.assertFalse(result['storage_plan_found']);self.assertEqual((result['context'],result['slots']),(32768,4));self.assertNotIn('plan',result)
        for options in (dict(context=True),dict(slots=0),dict(chunk_size=4097),dict(frame_bytes=255),dict(max_cache_bytes=2**31+1)):
            with self.assertRaises(ValueError):plan_reference_stages(layout,[2**30]*2,**options)
        with self.assertRaises(ValueError):plan_reference_stages(layout,[True,2**30])
        with self.assertRaises(ValueError):plan_reference_stages(layout,[2**30])
        with self.no_payload(),self.assertRaises(ValueError):plan_reference_stages(layout,[2**30]*2,projection_workspace_bytes=1)

    def test_header_drift_and_unowned_header_failure_are_not_capacity_results(self):
        import dllm.reference_stage_plan as planner
        for family in ('glm','flash','deepseek'):
            path,_,_=self.snapshot(family);layout=inspect_reference_storage(path,family)
            source=path/'config.json';replacement=path/'new';replacement.write_bytes(source.read_bytes());replacement.replace(source)
            with self.assertRaisesRegex(ValueError,'changed'):plan_reference_stages(layout,[2**30]*2)
            original=planner.read_header
            def drift(*args,**kwargs):
                result=original(*args,**kwargs);replacement.write_bytes(source.read_bytes());replacement.replace(source);return result
            with patch.object(planner,'read_header',drift),self.assertRaisesRegex(ValueError,'changed'):inspect_reference_storage(path,family)
            module={'glm':'glm','flash':'glm_flash','deepseek':'deepseek'}[family]
            with patch('dllm.'+module+'_reference_loader.audit_headers',side_effect=ValueError('unowned head')):
                with self.no_payload(),self.assertRaisesRegex(ValueError,'unowned'):inspect_reference_storage(path,family)

    def test_header_pass_count_does_not_grow_with_candidates_and_config_is_detached(self):
        import dllm.reference_stage_plan as planner
        path,_,_=self.snapshot('deepseek')
        with self.no_payload(),patch.object(planner,'read_header',wraps=planner.read_header) as read:
            layout=inspect_reference_storage(path,'deepseek');count=read.call_count
            config=layout.config;config['num_hidden_layers']=99
            for budgets in ([2**30]*2,[2**30]*3):self.assertTrue(plan_reference_stages(layout,budgets)['storage_plan_found'])
            self.assertEqual(read.call_count,count)
        self.assertEqual(layout.config['num_hidden_layers'],3)

    def test_actual_dry_run_command_ranges_output_and_no_fit_preserve_existing_file(self):
        path,_,_=self.snapshot('deepseek');target=self.path/'plan.json'
        environment=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        args=[sys.executable,'-m','scripts.plan_reference_host_topology','deepseek',str(path),'--budget-mib','256','256','--context','128','--plan-output',str(target)]
        result=subprocess.run(args,capture_output=True,text=True,timeout=15,env=environment)
        self.assertEqual(result.returncode,0,result.stderr);report=json.loads(result.stdout)
        self.assertEqual(json.loads(target.read_text()),report['plan']);self.assertEqual(set(report['plan']),{'ranges'})
        before=target.read_bytes();result=subprocess.run(args+['--budget-mib','1','1'],capture_output=True,text=True,timeout=15,env=environment)
        self.assertEqual(result.returncode,1,result.stderr);self.assertFalse(json.loads(result.stdout)['storage_plan_found']);self.assertEqual(target.read_bytes(),before)
        with self.no_payload(),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            command.main(['deepseek',str(path),'--budget-mib','256','256','--plan-output',str(path/'config.json')])
        self.assertEqual(json.loads((path/'config.json').read_text())['model_type'],'deepseek_v4')

    def test_deepseek_integer_width_and_invalid_position_metadata_are_preserved(self):
        import struct
        from tests.test_packed_checkpoint import write_raw
        from dllm.deepseek_reference_loader import load_reference_stage
        path,config,headers=self.snapshot('deepseek');before=inspect_reference_storage(path,'deepseek')
        shard=path/'model.safetensors';raw=shard.read_bytes();base=8+struct.unpack('<Q',raw[:8])[0];payload=bytearray();added=0
        for entry in headers.values():
            start,end=entry['data_offsets'];value=raw[base+start:base+end]
            if entry['dtype']=='I32':
                value=bytes(torch.frombuffer(bytearray(value),dtype=torch.int32).long().view(torch.uint8).tolist())
                entry['dtype']='I64';added+=end-start
            start=len(payload);payload.extend(value);entry['data_offsets']=[start,len(payload)]
        write_raw(shard,headers,payload)
        layout=inspect_reference_storage(path,'deepseek')
        self.assertEqual(sum(v[0] for v in layout.components)-sum(v[0] for v in before.components),added)
        stage=load_reference_stage(path,0,1);self.assertEqual(stage.blocks[0].table.dtype,torch.int64);stage.cleanup()
        (path/'config.json').write_text(json.dumps(config|dict(max_position_embeddings='bad')))
        layout=inspect_reference_storage(path,'deepseek')
        with self.assertRaisesRegex(ValueError,'position'):plan_reference_stages(layout,[2**30]*2)
