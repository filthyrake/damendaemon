from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from dllm.qualification_scorecard import evaluate_scorecard,verify_evidence,METRICS
from scripts.evaluate_qualification_scorecard import main


def fixture():
    identity=dict(model_repository='fixture/model',model_revision='a'*40,checkpoint_sha256='b'*64,tokenizer_sha256='c'*64,
                  precision='fixture Q4/FP32',hardware='synthetic hardware label; no benchmark run',topology='single device',
                  context=32768,prompt_tokens=512,output_tokens=128,concurrency=4,prompts_sha256='d'*64,eos_policy='fixed output')
    refs=[dict(path='fixture.json',sha256=hashlib.sha256(b'{}').hexdigest())]
    gate=lambda method:dict(result='pass',scope='actual_hardware',method=method,identity=dict(identity),evidence=deepcopy(refs))
    def measure(phases):
        metrics={name:1. for name in METRICS};metrics.update(decode_tokens_per_second=100.,aggregate_tokens_per_second=400.,
                first_output_p50_seconds=10.,first_output_p95_seconds=20.,request_p50_seconds=30.,request_p95_seconds=40.,
                peak_allocated_bytes=1024,peak_reserved_bytes=2048)
        return dict(identity=dict(identity),engine_revision='e'*40,scope='actual_hardware',workload_passed=True,shapes_warmed=True,
                    decode_timing='engine_token_steps',warmup_rounds=dict.fromkeys(phases,3),measured_rounds=dict.fromkeys(phases,10),
                    metrics=metrics,evidence=deepcopy(refs))
    cell=dict(id='synthetic-cell',issue='https://github.com/filthyrake/dllm/issues/54',availability='implemented',identity=identity,
              implementation_commit='f'*40,numerical=gate('independent_reference'),lifecycle=gate('endpoint_lifecycle'),
              measurement=measure(('A1','A2')),comparators=[dict(engine='baseline',availability='measured',exact_text_match=True,measurement=measure(('B1','B2')))])
    return dict(version=1,cells=[cell])


def row(value):return evaluate_scorecard(value)['cells'][0]


class QualificationScorecardTests(unittest.TestCase):
    def test_per_cell_thresholds_and_no_averaging(self):
        document=fixture();metrics=document['cells'][0]['measurement']['metrics']
        metrics.update(decode_tokens_per_second=95.,aggregate_tokens_per_second=380.,first_output_p50_seconds=11.,first_output_p95_seconds=22.)
        self.assertTrue(row(document)['competitive_pass']);self.assertEqual(row(document)['state'],'competitive_pass')
        metrics['decode_tokens_per_second']=94.999
        self.assertFalse(row(document)['competitive_pass']);self.assertEqual(row(document)['state'],'performance_measured')
        other=deepcopy(fixture()['cells'][0]);other['id']='fast';document['cells'].append(other)
        self.assertFalse(evaluate_scorecard(document)['all_cells_competitive'])

    def test_missing_failed_and_fixture_gates_never_become_pass(self):
        for gate in ('numerical','lifecycle'):
            for change in (None,{'result':'fail'},{'scope':'cpu_fixture'},{'method':'mocked'}):
                document=fixture();record=document['cells'][0]
                record[gate]=None if change is None else record[gate]|change
                self.assertIsNone(row(document)['competitive_pass'])
                if change=={'result':'fail'}:self.assertFalse(row(document)[gate+'_pass'])
        document=fixture();cell=document['cells'][0];cell.pop('measurement');cell.pop('lifecycle');cell.pop('numerical')
        self.assertEqual(row(document)['state'],'implemented_unvalidated')
        self.assertIsNone(row(document)['numerical_pass']);self.assertIsNone(row(document)['competitive_pass'])

    def test_missing_metric_proxy_identity_and_rounds_cannot_qualify(self):
        for change in ('metric','proxy','rounds','identity','workload','fixture','shapes'):
            document=fixture();measure=document['cells'][0]['measurement']
            if change=='metric':measure['metrics']['decode_tokens_per_second']=None
            if change=='proxy':measure['decode_timing']='client SSE proxy'
            if change=='rounds':measure['measured_rounds']['A2']=9
            if change=='identity':measure['identity']['precision']='other arithmetic'
            if change=='workload':measure['workload_passed']=False
            if change=='fixture':measure['scope']='cpu_fixture'
            if change=='shapes':measure['shapes_warmed']=False
            result=row(document);self.assertIsNone(result['competitive_pass']);self.assertTrue(result['gate_notes']['measurement'])
        document=fixture();document['cells'][0]['comparators'][0]['measurement']['identity']['hardware']='different device'
        self.assertIsNone(row(document)['competitive_pass'])

    def test_unsupported_and_blocked_comparators_are_not_wins(self):
        document=fixture();cell=document['cells'][0];refs=cell['measurement']['evidence']
        cell['comparators']=[dict(engine='unsupported',availability='unsupported',reason='format unavailable',evidence=refs)]
        self.assertIsNone(row(document)['competitive_pass'])
        cell['comparators'].append(fixture()['cells'][0]['comparators'][0]);self.assertTrue(row(document)['competitive_pass'])
        cell['comparators'][0]['availability']='blocked';self.assertIsNone(row(document)['competitive_pass'])
        cell.update(availability='blocked',reason='hardware unavailable',availability_evidence=refs)
        self.assertEqual(row(document)['state'],'blocked');self.assertIsNone(row(document)['competitive_pass'])

    def test_exact_text_mismatch_is_retained_and_blocks_competitive_pass(self):
        for exact in (None,False):
            document=fixture();document['cells'][0]['comparators'][0]['exact_text_match']=exact
            result=row(document);self.assertIsNone(result['competitive_pass'])
            self.assertIs(result['comparators'][0]['exact_text_match'],exact)
        document=fixture();cell=document['cells'][0];cell.pop('measurement');cell.pop('lifecycle')
        self.assertEqual(row(document)['state'],'numerical_pass')
        cell['lifecycle']=fixture()['cells'][0]['lifecycle'];self.assertEqual(row(document)['state'],'lifecycle_pass')

    def test_unresolved_identity_is_explicit_and_cannot_qualify(self):
        document=fixture();cell=document['cells'][0]
        cell['identity']['checkpoint_sha256']=None
        cell['measurement']['identity']['checkpoint_sha256']=None
        cell['numerical']['identity']['checkpoint_sha256']=None
        cell['lifecycle']['identity']['checkpoint_sha256']=None
        cell['comparators'][0]['measurement']['identity']['checkpoint_sha256']=None
        result=row(document)
        self.assertIsNone(result['competitive_pass']);self.assertIsNone(result['numerical_pass'])
        self.assertEqual(result['state'],'implemented_unvalidated')
        self.assertIsNone(result['identity']['checkpoint_sha256'])

    def test_availability_report_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'fixture.json').write_bytes(b'{}');document=fixture();cell=document['cells'][0]
            cell.update(availability='blocked',reason='unavailable',availability_evidence=[dict(path='blocked.json',sha256='a'*64)])
            (root/'blocked.json').write_bytes(b'{}')
            evaluate_scorecard(document)
            with self.assertRaisesRegex(ValueError,'digest'):verify_evidence(document,root)

    def test_invalid_numbers_pins_duplicate_cells_and_metrics_fail(self):
        for value in (True,-1,float('nan'),float('inf'),10**400):
            document=fixture();document['cells'][0]['measurement']['metrics']['decode_tokens_per_second']=value
            with self.assertRaises(ValueError):evaluate_scorecard(document)
        document=fixture();document['cells'][0]['identity']['model_revision']='main'
        with self.assertRaises(ValueError):evaluate_scorecard(document)
        document=fixture();document['cells']*=2
        with self.assertRaises(ValueError):evaluate_scorecard(document)
        document=fixture();document['cells'][0]['measurement']['metrics']['peak_allocated_bytes']=4096
        with self.assertRaises(ValueError):evaluate_scorecard(document)

    def test_cli_rejects_oversized_metric_and_excessive_json_nesting(self):
        document=fixture()
        document['cells'][0]['measurement']['metrics']['decode_tokens_per_second']=10**400
        cases=[json.dumps(document),'{"version":1,"cells":'+'['*10000+'0'+']'*10000+'}']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.json'
            for index,raw in enumerate(cases):
                with self.subTest(case=index):
                    source.write_text(raw);out=io.StringIO();err=io.StringIO()
                    with redirect_stdout(out),redirect_stderr(err),self.assertRaises(SystemExit) as caught:
                        main([str(source),'--evidence-root',str(root)])
                    self.assertEqual(caught.exception.code,2)
                    self.assertEqual(out.getvalue(),'')
                    self.assertIn('Scorecard rejected:',err.getvalue())
                    self.assertNotIn('Traceback',err.getvalue())

    def test_bounded_cli_binds_hashes_and_rejects_missing_or_escaping_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.json';report=root/'fixture.json';report.write_bytes(b'{}');document=fixture();source.write_text(json.dumps(document))
            output=io.StringIO()
            with redirect_stdout(output):self.assertEqual(main([str(source),'--evidence-root',str(root)]),0)
            result=json.loads(output.getvalue());self.assertEqual(result['verified_evidence_files'],1)
            report.write_bytes(b'{"changed":true}')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main([str(source),'--evidence-root',str(root)])
            document['cells'][0]['measurement']['evidence'][0]['path']='../outside.json'
            with self.assertRaises(ValueError):verify_evidence(document,root)
            source.write_text('{"version":1,"version":1,"cells":[]}')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main([str(source),'--evidence-root',str(root)])
