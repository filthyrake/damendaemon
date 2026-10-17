from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from damendaemon.cuda_host_storage import audit_cuda_host_storage
from damendaemon.reference_stage_plan import inspect_reference_storage
from damendaemon.cuda_stage_placement import _recipe
from damendaemon import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader
from damendaemon.glm_stage_slots import GlmStageSlots
from damendaemon.glm_flash_stage_slots import FlashStageSlots
from damendaemon.deepseek_stage_slots import DeepSeekStageSlots
from tests import test_reference_stage_plan as prior,test_deepseek_reference_loader as deepseek
from scripts import audit_cuda_host_storage as command


class CudaHostStorageTests(unittest.TestCase):
    setUp=prior.ReferenceStoragePlanTests.setUp
    snapshot=prior.ReferenceStoragePlanTests.snapshot
    no_payload=prior.ReferenceStoragePlanTests.no_payload

    def audit(self,layout,ranges,**options):
        host=options.pop('host_budgets',[2**30]*len(ranges));device=options.pop('device_budgets',[2**30]*len(ranges))
        return audit_cuda_host_storage(layout,ranges,host,device,context=129,slots=2,**options)

    def test_header_accounting_matches_unique_prepared_payload_and_slot_reservation(self):
        for family,loader,kind,cuts in (('glm',glm_reference_loader,GlmStageSlots,[(0,2),(2,3)]),
                ('flash',glm_flash_reference_loader,FlashStageSlots,[(0,1),(1,2),(2,3)]),
                ('deepseek',deepseek_reference_loader,DeepSeekStageSlots,[(0,1),(1,2),(2,3)])):
            path,_,_=self.snapshot(family)
            with self.no_payload():layout=inspect_reference_storage(path,family);report=self.audit(layout,cuts)
            self.assertTrue(report['declared_limits_fit']);self.assertFalse(report['execution_admitted'])
            self.assertFalse(report['device_compatibility_verified']);self.assertFalse(report['peak_memory_measured'])
            for (start,end),item in zip(cuts,report['stages']):
                stage=loader.load_reference_stage(path,start,end);module,_=_recipe(stage,256*1024**2)
                actual=sum(t.numel()*t.element_size() for t in module._tensors(stage,stage=True).values())
                self.assertEqual(item['device_payload_bytes'],actual)
                slots=kind(stage,129,2);self.assertEqual(item['cache_reservation_bytes'],slots.reserved_cache_bytes);slots.close()
                self.assertEqual(item['derived_device_frequency_bytes'],128 if family=='deepseek' else 0)

    def test_shared_frequencies_count_once_per_owned_ratio_not_layer(self):
        config=deepseek.geometry()|dict(compress_ratios=[0,4,4])
        with patch.object(deepseek,'geometry',return_value=config):path,_,_=self.snapshot('deepseek')
        with self.no_payload():report=self.audit(inspect_reference_storage(path,'deepseek'),[(0,1),(1,3)])
        self.assertEqual([s['derived_device_frequency_bytes'] for s in report['stages']],[128,128])
        self.assertEqual([s['host_preparation_frequency_bytes'] for s in report['stages']],[256,256])
        stage=deepseek_reference_loader.load_reference_stage(path,1,3);module,_=_recipe(stage,256*1024**2)
        self.assertIs(stage.blocks[0].frequencies,stage.blocks[1].frequencies)
        self.assertEqual(report['stages'][1]['device_payload_bytes'],sum(t.numel()*t.element_size() for t in module._tensors(stage,stage=True).values()))
        stage.cleanup()

    def test_exact_and_below_budgets_and_caps_do_not_change_plan_or_context(self):
        path,_,_=self.snapshot('deepseek');layout=inspect_reference_storage(path,'deepseek');cuts=[(0,1),(1,3)]
        report=self.audit(layout,cuts)
        host=[s['required_declared_host_bytes'] for s in report['stages']]
        device=[s['required_declared_device_bytes'] for s in report['stages']]
        with self.no_payload():
            self.assertTrue(self.audit(layout,cuts,host_budgets=host,device_budgets=device)['declared_limits_fit'])
            for options,label in ((dict(host_budgets=[host[0]-1,host[1]]),'host_budget'),
                    (dict(device_budgets=[device[0]-1,device[1]]),'device_budget'),(dict(max_device_bytes=1),'device_payload'),
                    (dict(max_weight_bytes=1),'loader_weight'),(dict(max_cache_bytes=1),'cache'),
                    (dict(max_output_bytes=1),'output'),(dict(max_transfer_bytes=1),'transfer'),(dict(frame_bytes=256),'frame')):
                result=self.audit(layout,cuts,**options);self.assertFalse(result['declared_limits_fit'])
                self.assertIn(label,result['stages'][0]['exceeded_caps']);self.assertEqual(result['plan']['ranges'],[[0,1],[1,3]])
                self.assertEqual((result['context'],result['slots']),(129,2))

    def test_bad_ranges_controls_and_source_drift_are_errors(self):
        path,_,_=self.snapshot('glm');layout=inspect_reference_storage(path,'glm')
        for cuts in ([(0,1),(1,3)],[(0,2),(1,3)],[(0,2)],[(False,2),(2,3)]):
            with self.no_payload(),self.assertRaises(ValueError):self.audit(layout,cuts)
        for options in (dict(device_budgets=[True,2**30]),dict(host_budgets=[2**30]),dict(max_transfer_bytes=0),dict(frame_bytes=255)):
            with self.no_payload(),self.assertRaises(ValueError):self.audit(layout,[(0,2),(2,3)],**options)
        file=path/'config.json';replacement=path/'changed';replacement.write_bytes(file.read_bytes());replacement.replace(file)
        with self.no_payload(),self.assertRaisesRegex(ValueError,'changed'):self.audit(layout,[(0,2),(2,3)])

    def test_cli_preserves_ranges_and_distinguishes_no_fit_from_bad_manifest(self):
        path,_,_=self.snapshot('flash');plan=self.path/'plan.json';plan.write_text(json.dumps(dict(ranges=[[0,1],[1,3]])))
        before=plan.read_bytes();args=['flash',str(path),'--plan',str(plan),'--host-budget-mib','1024','1024',
            '--device-budget-mib','1024','1024','--context','128']
        environment=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        for extra,code in (([],0),(['--device-budget-mib','1','1'],1)):
            result=subprocess.run([sys.executable,'-m','scripts.audit_cuda_host_storage',*args,*extra],capture_output=True,text=True,timeout=15,env=environment)
            self.assertEqual(result.returncode,code,result.stderr);report=json.loads(result.stdout)
            self.assertEqual(report['declared_limits_fit'],code==0);self.assertEqual(plan.read_bytes(),before)
        with self.no_payload(),redirect_stdout(io.StringIO()):self.assertEqual(command.main(args),0)
        plan.write_text(json.dumps(dict(ranges=[[0,1],[1,3]],device='cuda:0')))
        with self.no_payload(),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):command.main(args)

    def test_image_preparation_includes_full_prompt_embedding_transfer(self):
        path,config,_=self.snapshot('flash');layout=inspect_reference_storage(path,'flash')
        with self.no_payload():
            text=self.audit(layout,[(0,1),(1,3)],chunk_size=1)
            images=self.audit(layout,[(0,1),(1,3)],chunk_size=1,input_kind='flash_embeddings')
        self.assertGreater(images['stages'][0]['required_transfer_payload_bytes'],text['stages'][0]['required_transfer_payload_bytes'])
        self.assertEqual(images['stages'][0]['required_output_bytes'],129*config['text_config']['hidden_size']*config['text_config']['hc_mult']*4)
        cap=text['stages'][0]['required_transfer_payload_bytes']
        with self.no_payload():result=self.audit(layout,[(0,1),(1,3)],chunk_size=1,input_kind='flash_embeddings',max_transfer_bytes=cap)
        self.assertIn('transfer',result['stages'][0]['exceeded_caps'])
