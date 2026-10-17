from contextlib import redirect_stderr,redirect_stdout
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from damendaemon.cpu_snapshot_identity import fingerprint
from damendaemon.glm_tool_text import GlmToolText
from damendaemon.glm_batch_generation import BatchReferenceGenerator
from damendaemon.glm_tool_generation import GlmToolReferenceGenerator
from scripts import run_glm_host_topology as cli,run_cpu_host_topology as shared
from tests import test_glm_host_topology_command as raw
from tests.test_glm_reference_loader import floating_control
from tests.test_glm_tools import TOOLS


class GlmNativeHostCommandTests(unittest.TestCase):
    setUp=raw.GlmHostTopologyCommandTests.setUp
    snapshot=raw.GlmHostTopologyCommandTests.snapshot
    launch=raw.GlmHostTopologyCommandTests.launch
    event=raw.GlmHostTopologyCommandTests.event
    workers=raw.GlmHostTopologyCommandTests.workers

    def args(self,role,index=0,mode='fp8_reference',extra=()):
        args=raw.GlmHostTopologyCommandTests.args(self,role,index,mode)
        if role=='complete':
            offset=args.index('--batch-inputs');del args[offset:offset+2]
        args+=['--native-chat','--context','2048','--context-limit','2048','--prefill-chunk','256','--max-new-tokens','3']
        return args+list(extra)

    def complete(self,extra):
        return subprocess.run([sys.executable,'-m','scripts.run_glm_host_topology',*self.args('complete',extra=extra)],
            capture_output=True,text=True,timeout=25,env=self.environment)

    def finish_workers(self,workers,code=0):
        for worker in workers:
            output,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,code,error)
            self.assertEqual('"event": "stopped"' in output,code==0)

    @staticmethod
    def row_events(events):
        result={}
        for event in events:
            if 'event' in event:continue
            event=dict(event);row=event.pop('row',0);result.setdefault(row,[]).append(event)
        return result

    def test_native_single_ragged_and_tools_match_independent_control_over_processes(self):
        messages=[dict(role='user',content='Hi')];rows=[messages,[dict(role='user',content='A longer prompt')]]
        history=self.path/'messages.json';history.write_text(json.dumps(rows))
        tools=self.path/'tools.json';tools.write_text(json.dumps(TOOLS))
        for mode in ('single','ragged','tools'):
            self.snapshot(4 if mode=='tools' else 3)
            text=GlmToolText(self.path,vocab_size=self.config['vocab_size'])
            control_type=GlmToolReferenceGenerator if mode=='tools' else BatchReferenceGenerator
            control=control_type(floating_control(self.config,self.values),text,max_context=2048)
            identity=fingerprint(self.path).model_id;common=['--verify-model-id','--model-id',identity]
            workers=self.workers('fp8_reference',common)
            request=['--prompt','Hi'] if mode=='single' else ['--batch-messages',str(history),'--ragged-batch']
            if mode=='tools':request+=['--tools',str(tools)]
            result=self.complete(common+request);self.assertEqual(result.returncode,0,result.stderr)
            self.finish_workers(workers)
            events=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(events[0]['native_chat']);self.assertTrue(events[0]['model_identity_verified'])
            if mode=='single':expected=list(control.generate(messages,max_new_tokens=3))
            elif mode=='tools':expected=list(control.generate_batch_tools(rows,TOOLS,max_new_tokens=3,ragged=True))
            else:expected=list(control.generate_batch(rows,max_new_tokens=3,ragged=True))
            self.assertEqual(self.row_events(events),self.row_events(expected))
            self.assertTrue(all('finish_reason' in e for e in events[-(1 if mode=='single' else 2):]))

    def test_raw_native_policy_disagreement_rejects_handshake(self):
        workers=self.workers('fp8_reference')
        args=self.args('complete',extra=['--prompt','abc']);args.remove('--native-chat')
        result=subprocess.run([sys.executable,'-m','scripts.run_glm_host_topology',*args],capture_output=True,text=True,timeout=20,env=self.environment)
        self.assertEqual(result.returncode,2,result.stderr);self.assertNotIn('finish_reason',result.stdout)
        self.finish_workers(workers,2)

    def test_bad_native_inputs_fail_before_connections_or_loading(self):
        self.peers.write_text(json.dumps([dict(stage=i,address='127.0.0.1',port=1234+i,
            server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]))
        invalid=self.path/'invalid.json';invalid.write_text('null')
        rows=self.path/'rows.json';rows.write_text(json.dumps([[dict(role='user',content='Hi')]]*3))
        options=[[],['--messages',str(invalid)],['--prompt','Hi','--tools',str(invalid)],
            ['--prompt','Hi','--ragged-batch'],['--batch-messages',str(rows)],
            ['--prompt','Hi','--max-new-tokens','2048'],['--tokens',str(invalid)]]
        for extra in options:
            with self.subTest(extra=extra),patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:cli.main(self.args('complete',extra=extra))
                self.assertEqual(error.exception.code,2)
        args=self.args('complete',extra=['--messages',str(invalid)]);args.remove('--native-chat')
        with patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(args)
        with patch.object(shared.socket,'socket',side_effect=AssertionError('bind')),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            cli.main(self.args('worker',1,extra=['--tools',str(invalid)]))

    def test_failed_stop_withholds_command_success_finals(self):
        workers=self.workers('fp8_reference');output=io.StringIO()
        from damendaemon.host_pipeline_worker import HostPipelineCoordinator
        with patch.object(HostPipelineCoordinator,'stop',side_effect=RuntimeError('STOP failed')),redirect_stdout(output),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:cli.main(self.args('complete',extra=['--prompt','Hi']))
        self.assertEqual(error.exception.code,2);self.assertNotIn('finish_reason',output.getvalue())
        self.finish_workers(workers,2)
