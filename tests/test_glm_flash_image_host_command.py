from contextlib import redirect_stdout,redirect_stderr
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from damendaemon.cpu_snapshot_identity import fingerprint
from damendaemon.glm_flash_image_tool_text import FlashImageToolText
from damendaemon.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from damendaemon.glm_flash_reference_loader import load_reference_model
from damendaemon.glm_flash_vision_loader import load_reference_vision
from damendaemon.glm_flash_inline_images import decode_inline_images
from scripts import run_glm_flash_host_topology as cli,run_cpu_host_topology as shared,flash_host_images
from tests import test_glm_flash_native_host_command as prior
from tests.test_glm_flash_inline_images import snapshot,messages
from tests.test_glm_flash_tool_generation import TOOLS


class FlashImageHostCommandTests(unittest.TestCase):
    setUp=prior.FlashNativeHostCommandTests.setUp
    event=prior.FlashNativeHostCommandTests.event
    workers=prior.FlashNativeHostCommandTests.workers
    launch=prior.FlashNativeHostCommandTests.launch
    complete=prior.FlashNativeHostCommandTests.complete
    finish_workers=prior.FlashNativeHostCommandTests.finish_workers

    def snapshot(self,count):
        self.count=count;snapshot(self.path);self.config=json.loads((self.path/'config.json').read_text())
        self.plan.write_text(json.dumps(dict(ranges=[[0,1],[1,3]] if count==2 else [[0,1],[1,2],[2,3]])))
        self.history=self.path/'messages.json';self.history.write_text(json.dumps(messages()))
        self.tools=self.path/'tools.json';self.tools.write_text(json.dumps(TOOLS))

    def args(self,role,index=0,mode='fp8_reference',extra=()):
        return prior.FlashNativeHostCommandTests.args(self,role,index,mode,extra)+['--flash-images']

    def test_two_three_process_image_and_tool_requests_match_full_model(self):
        for count,tools in ((2,False),(3,True)):
            self.snapshot(count);identity=fingerprint(self.path).model_id
            common=['--verify-model-id','--model-id',identity]
            workers=self.workers('fp8_reference',common)
            result=self.complete(common+['--messages',str(self.history)]+(['--tools',str(self.tools)] if tools else []))
            self.assertEqual(result.returncode,0,result.stderr);self.finish_workers(workers)
            events=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(events[0]['modality'],'still_image');self.assertTrue(events[0]['model_identity_verified'])
            self.assertEqual(len(events[0]['stage_layers']),count)
            text=FlashImageToolText(self.path,vocab_size=self.config['text_config']['vocab_size'])
            control=FlashImageToolReferenceGenerator(load_reference_model(self.path),text,load_reference_vision(self.path),max_context=2048)
            decoded=decode_inline_images(messages(),tools=TOOLS if tools else None)
            expected=list(control.generate_image_tools(decoded.messages,decoded.images,TOOLS,max_new_tokens=3) if tools
                          else control.generate_images(decoded.messages,decoded.images,max_new_tokens=3))
            self.assertEqual(events[1:],expected);self.assertIn('finish_reason',events[-1])

    def test_invalid_request_mode_processor_and_budgets_precede_network_and_payloads(self):
        self.peers.write_text(json.dumps([dict(stage=i,address='127.0.0.1',port=1234+i,
            server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]))
        bad=self.path/'bad.json';bad.write_text('null')
        options=[[],['--prompt','Hi'],['--batch-messages',str(self.history)],['--messages',str(bad)],
            ['--messages',str(self.history),'--tools',str(bad)],['--messages',str(self.history),'--max-patches','1'],
            ['--messages',str(self.history),'--max-new-tokens','2048'],['--messages',str(self.history),'--vision-weight-mib','2049']]
        with patch.object(shared,'connect_peer',side_effect=AssertionError('network before preflight')),patch.object(flash_host_images,'load_vision',side_effect=AssertionError('payload before preflight')):
            for extra in options:
                with self.subTest(extra=extra),redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:cli.main(self.args('complete',extra=extra))
                    self.assertEqual(error.exception.code,2)
            processor=self.path/'processor_config.json';original=processor.read_text();processor.write_text('{}')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(self.args('complete',extra=['--messages',str(self.history)]))
            processor.write_text(original)
            args=self.args('complete',extra=['--messages',str(self.history),'--max-patches','64']);args.remove('--flash-images')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(args)
        for extra in (['--vision-weight-mib','256'],['--messages',str(self.history)]):
            with patch.object(shared.socket,'socket',side_effect=AssertionError('worker bind')),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                cli.main(self.args('worker',1,extra=extra))
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(self.args('serve',extra=['--messages',str(self.history)]))

    def test_vision_load_finishes_before_first_connection(self):
        self.snapshot(2);workers=self.workers('fp8_reference');order=[]
        load=flash_host_images.load_vision;connect=shared.connect_peer
        def vision(*args):
            result=load(*args);order.append('vision loaded');return result
        def peer(*args,**kwargs):
            self.assertEqual(order,['vision loaded']);order.append('connected');return connect(*args,**kwargs)
        with patch.object(flash_host_images,'load_vision',vision),patch.object(shared,'connect_peer',peer),redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(self.args('complete',extra=['--messages',str(self.history)])),0)
        self.assertEqual(order,['vision loaded','connected']);self.finish_workers(workers)

    def test_image_text_policy_mismatch_has_no_success_final(self):
        self.snapshot(2);workers=self.workers('fp8_reference')
        args=self.args('complete',extra=['--prompt','Hi']);args.remove('--flash-images')
        result=subprocess.run([sys.executable,'-m','scripts.run_glm_flash_host_topology',*args],capture_output=True,text=True,timeout=20,env=self.environment)
        self.assertEqual(result.returncode,2,result.stderr);self.assertNotIn('finish_reason',result.stdout);self.finish_workers(workers,2)

    def test_failed_stop_withholds_image_completion_success(self):
        self.snapshot(2);workers=self.workers('fp8_reference');output=io.StringIO()
        from damendaemon.host_pipeline_worker import HostPipelineCoordinator
        with patch.object(HostPipelineCoordinator,'stop',side_effect=RuntimeError('STOP failed')),redirect_stdout(output),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(self.args('complete',extra=['--messages',str(self.history)]))
        self.assertEqual(error.exception.code,2);self.assertNotIn('finish_reason',output.getvalue());self.finish_workers(workers,2)
