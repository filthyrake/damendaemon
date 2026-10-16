"""Pure report validation; never launch the hardware runner."""
import subprocess
import sys
import unittest
from pathlib import Path

HARNESS=r'''
from copy import deepcopy
from scripts.compare_rerank_a2 import compare_reports
import sys
case=sys.argv[1]
report={'query':'q','base_documents':['a','b'],'model':'fixture','precision':'float32',
        'gpu_uuid':'fixture','rows':[]}
for count in (1,4,16,64):
    for concurrency in (1,4):
        request={'results':[{'index':i,'relevance_score':-.25*i} for i in range(count)],
                 'usage':{'total_tokens':count*2}}
        report['rows'].append({'pairs_per_request':count,'concurrency':concurrency,
            'trials':[{'warmup':index==0,'requests':[deepcopy(request) for _ in range(concurrency)]}
                      for index in range(6)]})
other=deepcopy(report)
request=report['rows'][2]['trials'][-1]['requests'][0]
results=other['rows'][2]['trials'][0]['requests'][0]['results']
if case in ('nan','inf','bool-score','text-score','huge-score'):
    results[1]['relevance_score']={'nan':float('nan'),'inf':float('inf'),'bool-score':True,
                                 'text-score':'-.25','huge-score':10**400}[case]
if case in ('bool-index','float-index','negative-index','range-index','duplicate-index'):
    results[1]['index']={'bool-index':True,'float-index':1.0,'negative-index':-1,
                        'range-index':4,'duplicate-index':0}[case]
if case=='short':results.pop()
if case=='extra':results.append(dict(results[0]))
if case=='null':other['rows'][2]['trials'][0]['requests'][0]['results']=None
if case=='nonobject':results[1]=[]
if case=='valid':results.reverse()
if case=='reference-nan':request['results'][1]['relevance_score']=float('nan')
for value in (report,other):
    if case=='empty-rows':value['rows']=[]
    if case=='missing-row':value['rows'].pop()
    if case=='duplicate-row':value['rows'][1]=deepcopy(value['rows'][0])
    if case=='row-order':value['rows'].reverse()
    if case=='short-trials':value['rows'][0]['trials'].pop()
    if case=='empty-trials':value['rows'][0]['trials']=[]
    if case=='extra-trials':value['rows'][0]['trials'].append(deepcopy(value['rows'][0]['trials'][-1]))
    if case=='bad-warmup':value['rows'][0]['trials'][1]['warmup']=True
    if case=='integer-warmup':value['rows'][0]['trials'][0]['warmup']=1
    if case=='missing-requests':value['rows'][0]['trials'][0]['requests']=[]
    if case=='boolean-count':value['rows'][0]['pairs_per_request']=True
    if case=='boolean-concurrency':value['rows'][0]['concurrency']=True
    if case=='nonobject-row':value['rows'][0]=[]
    if case=='nonobject-trial':value['rows'][0]['trials'][0]=[]
result=compare_reports(report,other)
assert result['passed'] is True and result['requests_checked']==240
print('passed')
'''


class RerankComparisonTests(unittest.TestCase):
    def run_case(self,case):
        return subprocess.run([sys.executable,'-S','-c',HARNESS,case],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=10)

    def test_malformed_results_cannot_report_agreement(self):
        for case in ('nan','inf','bool-score','text-score','huge-score','bool-index','float-index',
                     'negative-index','range-index','duplicate-index','short','extra','null',
                     'nonobject','reference-nan'):
            with self.subTest(case=case):
                result=self.run_case(case)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,'')
                self.assertIn('ValueError',result.stderr)

    def test_complete_coverage_is_required_before_agreement(self):
        for case in ('empty-rows','missing-row','duplicate-row','row-order','short-trials',
                     'empty-trials','extra-trials','bad-warmup','integer-warmup',
                     'missing-requests','boolean-count','boolean-concurrency',
                     'nonobject-row','nonobject-trial'):
            with self.subTest(case=case):
                result=self.run_case(case)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,'')
                self.assertIn('ValueError',result.stderr)

    def test_reordered_finite_negative_scores_preserve_agreement(self):
        result=self.run_case('valid')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'passed\n')

    def test_bounded_strict_response_reader(self):
        script=r'''
import io,json
from unittest.mock import patch
from scripts import compare_rerank_a2 as comparator
valid=b'{"results":[{"index":0,"relevance_score":-0.2}],"usage":null}'
class Bounded(io.BytesIO):
    def read(self,size=-1):
        assert size==len(valid)+1, 'unbounded response read'
        return super().read(size)
with patch.object(comparator,'MAX_RESPONSE_BYTES',len(valid)):
    assert comparator._read_response(Bounded(valid),1)['results'][0]['index']==0
    try:comparator._read_response(Bounded(valid+b' '),1)
    except ValueError:pass
    else:raise AssertionError('oversized response accepted')
for error in ({},[],False,0,'','failure'):
    raw=json.dumps({'error':error,'results':[{'index':0,'relevance_score':0.5}]}).encode()
    try:comparator._read_response(io.BytesIO(raw),1)
    except ValueError:pass
    else:raise AssertionError('explicit server error accepted')
assert comparator._read_response(io.BytesIO(b'{"error":null,"results":[{"index":0,"relevance_score":0.5}]}'),1)['results']
for raw in (b'{"results":[],"results":[{"index":0,"relevance_score":0}]}',
            b'{"results":[{"index":1,"index":0,"relevance_score":0}]}',
            b'{"results":[{"index":0,"relevance_score":NaN}]}',
            b'{"results":[]}',b'[]',b'null',b'\xff',
            b'{"extra":'+b'['*2000+b'0'+b']'*2000+b'}'):
    try:comparator._read_response(io.BytesIO(raw),1)
    except ValueError:pass
    else:raise AssertionError('invalid response accepted')
'''
        result=subprocess.run([sys.executable,'-S','-c',script],cwd=Path(__file__).resolve().parents[1],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
