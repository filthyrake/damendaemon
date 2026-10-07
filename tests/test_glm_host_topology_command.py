import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
import torch
from dllm.cpu_reference_text import CpuReferenceText
from dllm.cpu_snapshot_identity import fingerprint
from scripts import run_glm_host_topology as cli,run_cpu_host_topology as shared
from tests import test_cpu_host_topology_command as prior
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture,floating_control
from tests.test_glm_checkpoint import config as geometry
from tests.tls_credentials import credentials


class GlmHostTopologyCommandTests(unittest.TestCase):
    event=prior.CpuHostTopologyCommandTests.event
    workers=prior.CpuHostTopologyCommandTests.workers

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)
        self.cert=credentials();self.plan=self.path/'plan.json';self.peers=self.path/'peers.json'
        self.batch=self.path/'batch.json'
        self.batch.write_text(json.dumps([dict(prompt='ab',max_new_tokens=3),dict(tokens=[3],max_new_tokens=1)]))
        self.environment={k:v for k,v in os.environ.items() if not k.startswith('DLLM_')}
        self.environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.fixture=SimpleNamespace(batch=self.batch,environment=self.environment)
        self.snapshot(3)

    def snapshot(self,count):
        self.count=count;text=text_fixture(self.path)
        config=geometry()|dict(num_hidden_layers=count+1,indexer_types=['full','shared']+['full']*(count-1),
                               mlp_layer_types=['dense']+['sparse']*count)
        with patch('tests.test_glm_reference_loader.geometry',return_value=config):
            self.config,_,self.values=fixture(self.path,vocab_size=text.vocab_size)
        ranges=[[0,2]]+[[i,i+1] for i in range(2,count+1)]
        self.plan.write_text(json.dumps({'ranges':ranges}))

    def args(self,role,index=0,mode='fp8_reference',extra=()):
        return prior.CpuHostTopologyCommandTests.args(self,role,index,mode,extra)

    def launch(self,args):
        process=subprocess.Popen([sys.executable,'-m','scripts.run_glm_host_topology',*args],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=self.environment)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup);return process

    def complete(self,extra=()):
        return subprocess.run([sys.executable,'-m','scripts.run_glm_host_topology',*self.args('complete',extra=extra)],
            capture_output=True,text=True,timeout=20,env=self.environment)

    def expected(self):
        text=CpuReferenceText(self.path,self.config);model=floating_control(self.config,self.values)
        rows=[]
        for item in json.loads(self.batch.read_text()):
            maximum=item['max_new_tokens'];tokens=text.tokens(item.get('prompt',item.get('tokens')),
                max_context=16,max_new_tokens=maximum,chunk_size=2)
            logits,state=model.prefill(torch.tensor([tokens]),capacity=len(tokens)+maximum)
            generated=[];reason='length'
            try:
                for step in range(maximum):
                    scores=logits[:,:text.vocab_size].clone()
                    for token in text.excluded_ids:scores[:,token]=-float('inf')
                    token=int(scores.argmax(-1));generated.append(token)
                    if token in text.eos_ids:reason='stop';break
                    if step+1<maximum:logits=model.forward(torch.tensor([[token]]),state,position=len(tokens)+step)
            finally:state.close()
            rows.append((generated,text.decode(generated),len(tokens),reason))
        return rows

    def test_three_four_process_verified_raw_completion_matches_independent_control(self):
        for count in (3,4):
            self.snapshot(count);identity=fingerprint(self.path).model_id
            extra=['--verify-model-id','--model-id',identity]
            workers=self.workers('fp8_reference',extra);result=self.complete(extra)
            self.assertEqual(result.returncode,0,result.stderr)
            for worker in workers:
                output,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,0,error)
                self.assertEqual(json.loads(output.splitlines()[-1])['event'],'stopped')
            events=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(events[0]['scope'],'glm_cpu_host_topology_reference')
            self.assertTrue(events[0]['model_identity_verified']);self.assertEqual(len(events[0]['stage_layers']),count)
            for row,(tokens,text,prompt,reason) in enumerate(self.expected()):
                self.assertEqual([e['token_id'] for e in events if e.get('row')==row and 'token_id' in e],tokens)
                final=next(e for e in events if e.get('row')==row and 'finish_reason' in e)
                self.assertEqual(final['text'],text);self.assertEqual(final['finish_reason'],reason)
                self.assertEqual(final['usage'],dict(prompt_tokens=prompt,completion_tokens=len(tokens),total_tokens=prompt+len(tokens)))
            self.assertTrue(all('finish_reason' in e for e in events[-2:]))

    def test_identity_verification_and_eos_policy_mismatch_have_no_success_final(self):
        for mode in ('source','verification','eos'):
            self.snapshot(3);identity=fingerprint(self.path).model_id
            worker_extra=['--verify-model-id','--model-id',identity] if mode=='verification' else []
            workers=self.workers('fp8_reference',worker_extra)
            if mode=='eos':(self.path/'generation_config.json').write_text(json.dumps({'eos_token_id':[1]}))
            extra=['--model-id',identity] if mode=='verification' else ['--model-id','cd'*32] if mode=='source' else []
            result=self.complete(extra);self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn('finish_reason',result.stdout)
            for worker in workers:
                output,error=worker.communicate(timeout=8);self.assertEqual(worker.returncode,2,error)
                self.assertNotIn('"event": "stopped"',output)

    def test_signalled_worker_and_partial_connection_failure_close_remaining_peer(self):
        workers=self.workers('fp8_reference');workers[-1].terminate()
        output,error=workers[-1].communicate(timeout=8)
        self.assertEqual(workers[-1].returncode,130,error);self.assertNotIn('"event": "stopped"',output)
        result=self.complete();self.assertEqual(result.returncode,2,result.stderr)
        self.assertNotIn('finish_reason',result.stdout)
        output,error=workers[0].communicate(timeout=8);self.assertEqual(workers[0].returncode,2,error)
        self.assertNotIn('"event": "stopped"',output)

    def test_unsupported_family_policy_and_bad_plan_fail_before_connections(self):
        self.peers.write_text(json.dumps([dict(stage=i,address='127.0.0.1',port=1234+i,
            server_name='localhost',peer_cert=str(self.cert/'server.pem')) for i in (1,2)]))
        for extra in (['--quantization','none'],['--chat-template','unused'],['--loader-workspace-mib','1'],
                      ['--model-id','bad'],['--verify-model-id']):
            with self.subTest(extra=extra),patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:cli.main(self.args('complete',extra=extra))
                self.assertEqual(failure.exception.code,2)
        self.plan.write_text(json.dumps({'ranges':[[0,1],[1,4]]}))
        with patch.object(shared,'connect_peer',side_effect=AssertionError('network')),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):cli.main(self.args('complete'))
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(self.args('serve'))
