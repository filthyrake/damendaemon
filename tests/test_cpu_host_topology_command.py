import io
import json
import select
import subprocess
import sys
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import Mock,patch
from urllib.request import Request,urlopen
from scripts import run_cpu_host_topology as cli
from scripts.run_cpu_pipeline_reference import main as local_main
from dllm.cpu_snapshot_identity import fingerprint
from tests import test_cpu_host_command as fixtures


class CpuHostTopologyCommandTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CpuHostCommandTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.path,self.cert=self.fixture.path,self.fixture.cert
        self.plan=self.path/'plan.json';self.peers=self.path/'peers.json';self.count=3
        self.plan.write_text(json.dumps({'ranges':[[0,1],[1,3],[3,4]]}))

    def args(self,role,index=0,mode='none',extra=()):
        worker=role=='worker'
        args=[role,str(self.path),'--plan',str(self.plan),'--index',str(index),
            '--ca',str(self.cert/'ca.pem'),'--cert',str(self.cert/('server.pem' if worker else 'client.pem')),
            '--key',str(self.cert/('server.key' if worker else 'client.key')),
            '--model-id','ab'*32,'--context','16','--slots','2','--prefill-chunk','2','--timeout','6','--quantization',mode]
        if worker:args+=['--address','127.0.0.1','--port','0','--peer-cert',str(self.cert/'client.pem')]
        else:args+=['--peers',str(self.peers)]
        if role=='complete':args+=['--batch-inputs',str(self.fixture.batch)]
        if role=='serve':args+=['--http-port','0','--heartbeat-interval','0.1']
        return args+list(extra)

    def launch(self,args):
        process=subprocess.Popen([sys.executable,'-m','scripts.run_cpu_host_topology',*args],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.fixture.environment)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup);return process

    def event(self,process):
        self.assertTrue(select.select([process.stdout],[],[],12)[0],'process did not announce readiness')
        line=process.stdout.readline()
        self.assertTrue(line,process.stderr.read() if process.poll() is not None else 'process closed output')
        return json.loads(line)

    def workers(self,mode='none',extra=()):
        processes=[];peers=[]
        for index in range(1,self.count):
            process=self.launch(self.args('worker',index,mode,extra));event=self.event(process)
            self.assertEqual(event['event'],'listening');processes.append(process)
            peers.append(dict(stage=index,address='127.0.0.1',port=event['port'],server_name='localhost',peer_cert=str(self.cert/'server.pem')))
        self.peers.write_text(json.dumps(peers));return processes

    def complete(self,mode='none',extra=()):
        return subprocess.run([sys.executable,'-m','scripts.run_cpu_host_topology',*self.args('complete',mode=mode,extra=extra)],
            capture_output=True,text=True,timeout=20,env=self.fixture.environment)

    def test_three_and_four_process_completion_matches_whole_snapshot(self):
        for mode,count in (('none',3),('q4_0',4),('q8_0',3)):
            self.count=count;self.plan.write_text(json.dumps({'ranges':[[0,1],[1,3],[3,4]] if count==3 else [[i,i+1] for i in range(4)]}))
            identity=fingerprint(self.path).model_id;extra=['--verify-model-id','--model-id',identity]
            workers=self.workers(mode,extra);result=self.complete(mode,extra)
            self.assertEqual(result.returncode,0,result.stderr)
            for worker in workers:
                output,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,0,error)
                self.assertEqual(json.loads(output.splitlines()[-1])['event'],'stopped')
            events=[json.loads(line) for line in result.stdout.splitlines()]
            output=io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(local_main([str(self.path),'--batch-inputs',str(self.fixture.batch),'--context','16',
                    '--stages','1','--quantization',mode,'--prefill-chunk','2']),0)
            expected=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[1:],expected[1:]);self.assertTrue(events[0]['model_identity_verified'])
            self.assertEqual(len(events[0]['stage_layers']),count)

    def test_http_sse_and_drain_stops_every_process(self):
        workers=self.workers('q4_0');coordinator=self.launch(self.args('serve',mode='q4_0',extra=['--stop-after-drain']))
        event=self.event(coordinator);self.assertEqual(event['event'],'http_ready');base=event['url']
        body=dict(model='model',prompt='hello world',max_tokens=3,stream=True,stream_options={'include_usage':True})
        with urlopen(Request(base+'/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=5) as response:wire=response.read().decode()
        self.assertIn('data: [DONE]',wire)
        events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        self.assertEqual(events[-1]['usage']['completion_tokens'],3)
        with urlopen(base+'/v1/models/status',timeout=3) as response:status=json.load(response)
        self.assertEqual(status['stage_layers'],[[0,1],[1,3],[3,4]])
        self.assertFalse(status['official_model_qualified'])
        with urlopen(Request(base+'/v1/scheduler/drain',b'{}',{'Content-Type':'application/json'}),timeout=3) as response:
            self.assertTrue(json.load(response)['draining'])
        for process in [coordinator,*workers]:
            output,error=process.communicate(timeout=8);self.assertEqual(process.returncode,0,error)
            self.assertEqual(json.loads(output.splitlines()[-1])['event'],'stopped')
        with self.assertRaises(OSError):urlopen(base+'/health',timeout=1)

    def test_contract_disagreement_and_worker_loss_fail_without_reuse(self):
        workers=self.workers();result=self.complete(extra=['--model-id','cd'*32])
        self.assertEqual(result.returncode,2,result.stderr);self.assertNotIn('finish_reason',result.stdout)
        for worker in workers:
            _,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,error)
        workers=self.workers();coordinator=self.launch(self.args('serve'))
        self.assertEqual(self.event(coordinator)['event'],'http_ready')
        workers[0].terminate()
        output,error=coordinator.communicate(timeout=8);self.assertEqual(coordinator.returncode,2,error)
        for worker in workers:
            _,error=worker.communicate(timeout=8);self.assertIn(worker.returncode,(2,130),error)
        self.assertNotIn('finish_reason',output)

    def test_signal_shutdown_discards_all_links_without_claiming_stop_ack(self):
        workers=self.workers();coordinator=self.launch(self.args('serve'))
        base=self.event(coordinator)['url'];coordinator.terminate()
        _,error=coordinator.communicate(timeout=8);self.assertEqual(coordinator.returncode,0,error)
        for worker in workers:
            output,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,error)
            self.assertNotIn('"event": "stopped"',output)
        with self.assertRaises(OSError):urlopen(base+'/health',timeout=1)

    def test_eos_and_verification_policy_mismatch_fail_session_setup(self):
        workers=self.workers()
        (self.path/'generation_config.json').write_text(json.dumps({'eos_token_id':[59]}))
        result=self.complete();self.assertEqual(result.returncode,2,result.stderr)
        self.assertNotIn('finish_reason',result.stdout)
        for worker in workers:
            _,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,error)
        identity=fingerprint(self.path).model_id
        workers=self.workers(extra=['--verify-model-id','--model-id',identity])
        result=self.complete(extra=['--model-id',identity]);self.assertEqual(result.returncode,2,result.stderr)
        self.assertNotIn('finish_reason',result.stdout)
        for worker in workers:
            _,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,error)

    def test_metadata_rejects_before_connection_or_tensor_loading(self):
        peers=[dict(stage=i,address='127.0.0.1',port=1234+i,server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]
        self.peers.write_text(json.dumps(peers))
        cases=[['--index','1'],['--model-id','bad'],['--slots','1'],['--timeout','nan'],['--prefill-chunk','0'],
               ['--address','127.0.0.1'],['--verify-model-id']]
        for extra in cases:
            with patch.object(cli,'connect_peer',side_effect=AssertionError('network')), \
                    patch.object(cli,'build_cpu_host_pipeline_worker',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:cli.main(self.args('complete',extra=extra))
                self.assertEqual(caught.exception.code,2)
        peers[1]['address']='hostname.invalid';self.peers.write_text(json.dumps(peers))
        with patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete'))
        self.plan.write_text(json.dumps({'ranges':[[0,1],[2,4]]}))
        with patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete'))

    def test_partial_connection_and_changed_verified_load_close_every_owner(self):
        peers=[dict(stage=i,address='127.0.0.1',port=1234+i,server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]
        self.peers.write_text(json.dumps(peers));transport=Mock()
        with patch.object(cli,'connect_peer',side_effect=[Mock(),OSError('second peer refused')]), \
                patch.object(cli,'HostTensorChannel',return_value=transport),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete'))
        transport.close.assert_called_once()
        identity=fingerprint(self.path).model_id;worker=Mock();transports=[Mock(),Mock()]
        def changed(*args,**kwargs):
            config=self.path/'config.json';config.write_text(config.read_text()+' ');return worker
        with patch.object(cli,'connect_peer',return_value=Mock()),patch.object(cli,'HostTensorChannel',side_effect=transports), \
                patch.object(cli,'build_cpu_host_pipeline_worker',side_effect=changed),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete',extra=['--verify-model-id','--model-id',identity]))
        worker.handshake.assert_not_called();worker.close.assert_called_once()
        for transport in transports:transport.close.assert_called_once()
