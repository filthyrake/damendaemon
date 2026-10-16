import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer,models,pre_tokenizers
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.cpu_pipeline_build import build_cpu_pipeline,CpuPipelineBuildError
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_reference_text import CpuReferenceText
from dllm.hybrid_pipeline import HybridPipeline,PipelineCancelled
from scripts.run_cpu_pipeline_reference import main
from tests.test_cpu_hybrid_stage import model
from tests.test_cpu_hybrid_checkpoint import snapshot

OPTIONS=dict(max_workspace_bytes=2**20,projection_workspace_bytes=2**20,max_output_bytes=65536)


def tokenizer(path):
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,'world':2,**{f'w{i}':i for i in range(3,60)}},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(path/'tokenizer.json'))
    (path/'generation_config.json').write_text(json.dumps({'eos_token_id':[]}))


class CpuCompletionTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def runner(self,source=None,**options):
        result=CpuPipelineReferenceGenerator(source or model(),[2**22]*2,vocab_size=60,**OPTIONS,**options)
        self.addCleanup(result.cleanup);return result

    def test_checkpoint_cli_matches_independent_greedy_steps(self):
        for mode in ('none','q4_0','q8_0'):
            snapshot(self.path);tokenizer(self.path)
            loaded=load_cpu_checkpoint(self.path,max_context=16,quantization=mode)
            pipeline=build_cpu_pipeline(loaded,[2**22]*2,max_batch=1,**OPTIONS)
            try:
                logits=pipeline.prefill([1,2]);expected=[]
                for i in range(4):
                    token=int(logits[0,:60].argmax());expected.append(token)
                    if i<3:logits,_=pipeline.decode(token)
            finally:pipeline.close()
            out=io.StringIO()
            with patch('sys.stdout',out),patch('dllm.device.selected_accelerator',side_effect=AssertionError('accelerator')):
                self.assertEqual(main([str(self.path),'--prompt','hello world','--context','16','--max-new-tokens','4',
                                       '--prefill-chunk','1','--stages','2','--quantization',mode]),0)
            events=[json.loads(line) for line in out.getvalue().splitlines()]
            self.assertFalse(events[0]['official_model_qualified'])
            self.assertEqual([e['token_id'] for e in events if 'token_id' in e],expected)
            self.assertEqual(events[-1]['usage'],dict(prompt_tokens=2,completion_tokens=4,total_tokens=6))
            self.assertEqual(events[-1]['text'],CpuReferenceText(self.path,loaded.source_config).decode(expected))

    def test_padded_and_media_logits_suppressed_eos_cleanup_before_success(self):
        runner=self.runner(eos_ids=[7],excluded_ids=[8]);original=HybridPipeline.prefill_chunk;pipelines=[]
        def controlled(pipeline,*args,**kwargs):
            original(pipeline,*args,**kwargs);pipelines.append(pipeline)
            logits=torch.zeros(1,64);logits[0,63]=100;logits[0,8]=90;logits[0,7]=80;return logits
        with patch.object(HybridPipeline,'prefill_chunk',controlled):
            stream=runner.generate([1,2],max_new_tokens=3);self.assertEqual(next(stream),{'token_id':7})
            self.assertFalse(pipelines[0].closed)
            final=next(stream);self.assertEqual(final['finish_reason'],'stop');self.assertEqual(final['usage']['completion_tokens'],1)
            self.assertTrue(pipelines[0].closed);self.assertIsNone(runner.pipeline)
            with self.assertRaises(StopIteration):next(stream)

    def test_invalid_inputs_and_precancel_do_not_build(self):
        runner=self.runner()
        with patch('dllm.cpu_pipeline_generation.build_cpu_pipeline',side_effect=AssertionError('built')):
            for tokens,options in (([60],{}),([True],{}),([],{}),([1],dict(max_new_tokens=16)),
                                   ([1],dict(chunk_size=0)),([1],dict(max_new_tokens=[2])),([1],dict(max_new_tokens=4097))):
                with self.assertRaises(ValueError):list(runner.generate(tokens,**(dict(max_new_tokens=2)|options)))
            with self.assertRaises(PipelineCancelled):list(runner.generate([1],max_new_tokens=2,cancelled=lambda:True))
        self.assertIsNone(runner.pipeline)
        for group in ([True],[1,True],[-1],[60]):
            with self.assertRaises(ValueError):self.runner(eos_ids=group)

    def test_cancelled_or_closed_stream_releases_and_reuses(self):
        for cancel in (False,True):
            runner=self.runner();stopped=[False]
            stream=runner.generate([1,2],max_new_tokens=3,cancelled=lambda:stopped[0]);next(stream);pipeline=runner.pipeline
            with self.assertRaisesRegex(RuntimeError,'active'):runner.cleanup()
            with self.assertRaisesRegex(RuntimeError,'active'):next(runner.generate([1],max_new_tokens=2))
            if cancel:
                stopped[0]=True
                with self.assertRaises(PipelineCancelled):next(stream)
            else:stream.close()
            self.assertTrue(pipeline.closed);self.assertIsNone(runner.pipeline)
            self.assertIn('finish_reason',list(runner.generate([1],max_new_tokens=1))[-1])

    def test_primary_decode_error_preserved_cleanup_retry_owned(self):
        runner=self.runner();stream=runner.generate([1,2],max_new_tokens=3);next(stream);pipeline=runner.pipeline
        with patch.object(pipeline,'decode_batch',side_effect=RuntimeError('primary decode')), \
             patch.object(pipeline,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary decode') as caught:next(stream)
        self.assertTrue(caught.exception.__notes__);self.assertIs(runner.pipeline,pipeline)
        runner.cleanup();self.assertTrue(pipeline.closed);self.assertIsNone(runner.pipeline)

    def test_invalid_logits_and_failed_success_cleanup_never_report_success(self):
        for shape,dtype,nonfinite in (((1,63),torch.float32,False),((1,64),torch.float16,False),((1,64),torch.float32,True)):
            runner=self.runner();value=torch.zeros(shape,dtype=dtype)
            if nonfinite:value[0,2]=float('nan')
            with patch.object(HybridPipeline,'prefill_chunk',return_value=value):
                with self.assertRaisesRegex(RuntimeError,'completion logits'):list(runner.generate([1],max_new_tokens=1))
            self.assertIsNone(runner.pipeline)
        runner=self.runner();stream=runner.generate([1],max_new_tokens=1);next(stream);pipeline=runner.pipeline
        with patch.object(pipeline,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'cleanup'):next(stream)
        self.assertIs(runner.pipeline,pipeline);runner.cleanup()

    def test_partial_construction_failure_retains_cleanup_owner(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        stage=SimpleNamespace(release_partial=Mock(side_effect=RuntimeError('release failure')))
        failure=CpuPipelineBuildError('partial construction',[stage]);runner=self.runner()
        with patch('dllm.cpu_pipeline_generation.build_cpu_pipeline',side_effect=failure):
            with self.assertRaises(CpuPipelineBuildError) as caught:list(runner.generate([1],max_new_tokens=1))
        self.assertIs(caught.exception,failure);self.assertIs(runner.pipeline,failure)
        self.assertFalse(failure.closed);stage.release_partial.side_effect=None
        runner.cleanup();self.assertTrue(failure.closed);self.assertIsNone(runner.pipeline)

    def test_cli_preflight_before_weight_payload(self):
        snapshot(self.path);tokenizer(self.path)
        base=[str(self.path),'--context','16','--max-new-tokens','2']
        token_file=self.path/'prompt.json'
        for invalid in ([],[60],[True],list(range(15)),{'messages':[]},'hello'):
            token_file.write_text(json.dumps(invalid))
            with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')),patch('sys.stderr',io.StringIO()):
                with self.assertRaises(SystemExit) as caught:main(base+['--tokens',str(token_file)])
                self.assertEqual(caught.exception.code,2)
        (self.path/'generation_config.json').write_text('{"eos_token_id": true}')
        with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):main(base+['--prompt','hello'])
