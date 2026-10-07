import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr,redirect_stdout
from unittest.mock import patch

from scripts import run_cpu_host_reference as cli
from scripts.run_cpu_pipeline_reference import main as local_main
from tests.test_cpu_hybrid_checkpoint import snapshot
from tests.test_cpu_pipeline_generation import tokenizer
from tests.tls_credentials import credentials


class CpuHostCommandTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);snapshot(self.path);tokenizer(self.path)
        self.cert=credentials()
        self.batch=self.path/'batch.json'
        self.batch.write_text(json.dumps([{'prompt':'hello world','max_new_tokens':3},
                                          {'tokens':[3],'max_new_tokens':1}]))
        self.environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def args(self,role,port=0,mode='none'):
        worker=role=='worker'
        return [role,str(self.path),'--address','127.0.0.1','--port',str(port),
            '--ca',str(self.cert/'ca.pem'),'--cert',str(self.cert/('server.pem' if worker else 'client.pem')),
            '--key',str(self.cert/('server.key' if worker else 'client.key')),
            '--peer-cert',str(self.cert/('client.pem' if worker else 'server.pem')),
            '--model-id','ab'*32,'--split','2','--context','16','--slots','2',
            '--timeout','4','--quantization',mode]+([] if worker else
                ['--server-name','localhost','--batch-inputs',str(self.batch),'--prefill-chunk','1'])

    def start_worker(self,mode='none',extra=()):
        process=subprocess.Popen([sys.executable,'-m','scripts.run_cpu_host_reference',*self.args('worker',mode=mode),*extra],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.environment)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup)
        self.assertTrue(select.select([process.stdout],[],[],10)[0],'worker never announced listener')
        line=process.stdout.readline();self.assertTrue(line,'worker exited before listener')
        event=json.loads(line);self.assertEqual(event['event'],'listening');self.assertFalse(event['official_model_qualified'])
        return process,event['port']

    def test_two_process_ragged_completions_match_whole_command(self):
        for mode in ('none','q4_0','q8_0'):
            worker,port=self.start_worker(mode)
            complete=subprocess.run([sys.executable,'-m','scripts.run_cpu_host_reference',*self.args('complete',port,mode)],
                capture_output=True,text=True,timeout=15,env=self.environment)
            worker_output,worker_error=worker.communicate(timeout=8)
            self.assertEqual(complete.returncode,0,complete.stderr)
            self.assertEqual(worker.returncode,0,worker_error)
            self.assertEqual(json.loads(worker_output.splitlines()[-1]),{'event':'stopped'})
            events=[json.loads(line) for line in complete.stdout.splitlines()]
            output=io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(local_main([str(self.path),'--batch-inputs',str(self.batch),'--context','16',
                    '--stages','2','--quantization',mode,'--prefill-chunk','1']),0)
            expected=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[1:],expected[1:])
            self.assertEqual(events[0]['event'],'ready');self.assertFalse(events[0]['official_model_qualified'])
            # Worker exited cleanly only after STOP; completion finals follow STOP acknowledgment.
            self.assertEqual([e['usage']['completion_tokens'] for e in events if 'finish_reason' in e],[3,1])

    def test_wrong_model_identity_or_pin_fails_without_completion(self):
        for extra in (['--model-id','cd'*32],['--peer-cert',str(self.cert/'client.pem')]):
            worker,port=self.start_worker()
            complete=subprocess.run([sys.executable,'-m','scripts.run_cpu_host_reference',*self.args('complete',port),*extra],
                capture_output=True,text=True,timeout=15,env=self.environment)
            _,worker_error=worker.communicate(timeout=8)
            self.assertEqual(complete.returncode,2,complete.stderr);self.assertEqual(worker.returncode,2,worker_error)
            self.assertNotIn('finish_reason',complete.stdout)
            self.assertNotIn('PRIVATE KEY',complete.stderr+worker_error)

    def test_preflight_rejects_before_network_or_payload(self):
        extra_cases=[['--model-id','bad'],['--split','4'],['--slots','1'],['--context','65'],
            ['--timeout','nan'],['--frame-mib','65'],['--weight-mib','2049'],['--context','513'],
            ['--address','hostname.invalid'],['--server-name',''],['--max-new-tokens','0']]
        for extra in extra_cases:
            with (self.subTest(extra=extra),patch.object(cli,'connect_peer',side_effect=AssertionError('network')),
                 patch.object(cli,'build_cpu_host_worker',side_effect=AssertionError('payload')),redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as caught:cli.main(self.args('complete',1234)+extra)
                self.assertEqual(caught.exception.code,2)
        self.batch.write_text(json.dumps([{'tokens':[999]}]))
        with patch.object(cli,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete',1234))

    def test_worker_accept_timeout_is_bounded(self):
        worker,_=self.start_worker(extra=['--timeout','0.2'])
        output,error=worker.communicate(timeout=5)
        self.assertEqual(worker.returncode,2,error);self.assertNotIn('ready',output)

    def test_single_prompt_and_tokenizer_identity_mismatch(self):
        worker,port=self.start_worker()
        args=self.args('complete',port)
        i=args.index('--batch-inputs');args[i:i+2]=['--prompt','hello world']
        complete=subprocess.run([sys.executable,'-m','scripts.run_cpu_host_reference',*args,'--max-new-tokens','2'],
            capture_output=True,text=True,timeout=15,env=self.environment)
        worker.communicate(timeout=8)
        self.assertEqual(complete.returncode,0,complete.stderr)
        events=[json.loads(line) for line in complete.stdout.splitlines()]
        self.assertNotIn('row',events[-1]);self.assertEqual(events[-1]['usage']['completion_tokens'],2)
        worker,port=self.start_worker()
        # Worker captured EOS before announcing its listener. The coordinator must
        # not silently use a different stopping contract with that prepared peer.
        (self.path/'generation_config.json').write_text(json.dumps({'eos_token_id':[59]}))
        complete=subprocess.run([sys.executable,'-m','scripts.run_cpu_host_reference',*self.args('complete',port)],
            capture_output=True,text=True,timeout=15,env=self.environment)
        worker.communicate(timeout=8)
        self.assertEqual(complete.returncode,2);self.assertEqual(worker.returncode,2)
        self.assertNotIn('finish_reason',complete.stdout)
