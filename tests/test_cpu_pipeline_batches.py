import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator
from dllm.cpu_pipeline_build import build_cpu_pipeline
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.hybrid_pipeline import HybridPipeline,PipelineCancelled
from scripts.run_cpu_pipeline_reference import main
from tests.test_cpu_hybrid_stage import model
from tests.test_cpu_hybrid_checkpoint import snapshot
from tests.test_cpu_pipeline_generation import tokenizer,OPTIONS


class CpuBatchTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name)

    def runner(self,source=None,**options):
        result=CpuPipelineReferenceGenerator(source or model(),[2**22]*2,vocab_size=60,**OPTIONS,**options)
        self.addCleanup(result.cleanup);return result

    def test_ragged_quantized_batches_match_independent_pipeline_logits(self):
        rows=[[1,2,3,4],[5],[6,7]];limits=[2,4,3]
        for mode in ('none','q4_0','q8_0'):
            snapshot(self.path);source=load_cpu_checkpoint(self.path,quantization=mode,max_context=16)
            controls=[build_cpu_pipeline(source,[2**22]*2,max_batch=1,**OPTIONS) for _ in rows]
            initial=[]
            try:
                for row,tokens in enumerate(rows):initial.append(controls[row].prefill(tokens))
                original=HybridPipeline.decode_batch;seen=[]
                def compare(pipeline,tokens,slots,**kwargs):
                    actual=original(pipeline,tokens,slots,**kwargs)
                    expected=torch.cat([original(controls[slot],[token],[0])[0] for token,slot in zip(tokens,slots)])
                    torch.testing.assert_close(actual[0],expected,rtol=2e-4,atol=2e-6);seen.append(list(slots));return actual
                runner=self.runner(source)
                with patch.object(HybridPipeline,'decode_batch',compare):events=list(runner.generate_batch(rows,max_new_tokens=limits,chunk_size=2))
                self.assertEqual(seen,[[0,1,2],[1,2],[1]])
                finals=[e for e in events if 'finish_reason' in e]
                self.assertEqual([e['row'] for e in finals],[0,1,2])
                self.assertEqual([e['usage']['prompt_tokens'] for e in finals],[4,1,2])
                self.assertEqual([e['usage']['completion_tokens'] for e in finals],limits)
                first=[e['token_id'] for e in events[:3]]
                self.assertEqual(first,[int(value[0,:60].argmax()) for value in initial])
                self.assertIsNone(runner.pipeline)
            finally:
                for control in controls:control.close()

    def test_eos_and_limits_keep_original_slots_without_finished_decode(self):
        runner=self.runner(eos_ids=[7]);prefill=HybridPipeline.prefill_chunk;decode=HybridPipeline.decode_batch;seen=[]
        def start(pipeline,*args,**kwargs):
            value=prefill(pipeline,*args,**kwargs)
            if value is not None:
                value=torch.zeros_like(value);value[0,7 if kwargs['slot']==1 else 8]=1
            return value
        def step(pipeline,tokens,slots,**kwargs):
            seen.append(list(slots));value,ids=decode(pipeline,tokens,slots,**kwargs)
            value=torch.zeros_like(value);value[:,8]=1;return value,ids
        with patch.object(HybridPipeline,'prefill_chunk',start),patch.object(HybridPipeline,'decode_batch',step):
            events=list(runner.generate_batch([[1],[2,3],[4]],max_new_tokens=[2,4,3],chunk_size=1))
        self.assertEqual(seen,[[0,2],[2]])
        finals=events[-3:];self.assertEqual([e['finish_reason'] for e in finals],['length','stop','length'])
        self.assertEqual([e['usage']['completion_tokens'] for e in finals],[2,1,3])

    def test_whole_batch_preflight_before_build(self):
        runner=self.runner()
        with patch('dllm.cpu_pipeline_generation.build_cpu_pipeline',side_effect=AssertionError('built')):
            for rows,limits in (([],2),([[1]]*5,2),([[1],[60]],2),([[1],[2]],[1]),([[1],[2]],[1,16]),
                                ([[1],[2]],[1,True]),([[1],[]],2)):
                with self.assertRaises(ValueError):list(runner.generate_batch(rows,max_new_tokens=limits))

    def test_later_prefill_failure_and_batch_cancel_close_all_slots(self):
        original=HybridPipeline.prefill_chunk;pipelines=[]
        def fail(pipeline,*args,**kwargs):
            pipelines.append(pipeline);result=original(pipeline,*args,**kwargs)
            if kwargs['slot']==1:raise RuntimeError('later prefill')
            return result
        runner=self.runner()
        with patch.object(HybridPipeline,'prefill_chunk',fail):
            with self.assertRaisesRegex(RuntimeError,'later prefill'):list(runner.generate_batch([[1],[2]],max_new_tokens=2))
        self.assertTrue(all(p.closed for p in pipelines));self.assertIsNone(runner.pipeline)
        cancelled=[False];stream=runner.generate_batch([[1],[2,3]],max_new_tokens=3,cancelled=lambda:cancelled[0])
        next(stream);next(stream);pipeline=runner.pipeline;cancelled[0]=True
        with self.assertRaises(PipelineCancelled):next(stream)
        self.assertTrue(pipeline.closed);self.assertIsNone(runner.pipeline)
        stream=runner.generate_batch([[1],[2]],max_new_tokens=3);next(stream);pipeline=runner.pipeline;stream.close()
        self.assertTrue(pipeline.closed)

    def test_actual_batch_cli_and_late_bad_row_preflight(self):
        snapshot(self.path);tokenizer(self.path);file=self.path/'batch.json'
        rows=[dict(prompt='hello world',max_new_tokens=1),dict(tokens=[3],max_new_tokens=3),dict(prompt='hello',max_new_tokens=2)]
        file.write_text(json.dumps(rows));out=io.StringIO()
        args=[str(self.path),'--batch-inputs',str(file),'--context','16','--quantization','q4_0','--stages','2','--prefill-chunk','1']
        with patch('sys.stdout',out):self.assertEqual(main(args),0)
        events=[json.loads(line) for line in out.getvalue().splitlines()];finals=events[-3:]
        self.assertEqual([e['row'] for e in finals],[0,1,2]);self.assertEqual([e['usage']['completion_tokens'] for e in finals],[1,3,2])
        self.assertTrue(all(isinstance(e['text'],str) for e in finals))
        for bad in (dict(tokens=[60]),dict(prompt='hello',tokens=[1]),dict(prompt=['hello']),dict(tokens='hello'),
                    dict(prompt='hello',max_new_tokens=True),dict(messages=[])):
            file.write_text(json.dumps(rows+[bad]))
            with patch('dllm.cpu_hybrid_checkpoint._payload',side_effect=AssertionError('payload')),patch('sys.stderr',io.StringIO()):
                with self.assertRaises(SystemExit) as caught:main(args)
                self.assertEqual(caught.exception.code,2)
