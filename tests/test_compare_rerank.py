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
request={'results':[{'index':0,'relevance_score':.5},{'index':1,'relevance_score':-.25}],
         'usage':{'total_tokens':4}}
report={'query':'q','base_documents':['a','b'],'model':'fixture','precision':'float32',
        'gpu_uuid':'fixture','rows':[{'pairs_per_request':2,'concurrency':1,
        'trials':[{'requests':[request]}]}]}
other=deepcopy(report)
results=other['rows'][0]['trials'][0]['requests'][0]['results']
if case in ('nan','inf','bool-score','text-score','huge-score'):
    results[1]['relevance_score']={'nan':float('nan'),'inf':float('inf'),'bool-score':True,
                                 'text-score':'-.25','huge-score':10**400}[case]
if case in ('bool-index','float-index','negative-index','range-index','duplicate-index'):
    results[1]['index']={'bool-index':True,'float-index':1.0,'negative-index':-1,
                        'range-index':2,'duplicate-index':0}[case]
if case=='short':results.pop()
if case=='extra':results.append(dict(results[0]))
if case=='null':other['rows'][0]['trials'][0]['requests'][0]['results']=None
if case=='nonobject':results[1]=[]
if case=='valid':results.reverse()
if case=='reference-nan':request['results'][1]['relevance_score']=float('nan')
result=compare_reports(report,other)
assert result['passed'] is True and result['requests_checked']==2
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

    def test_reordered_finite_negative_scores_preserve_agreement(self):
        result=self.run_case('valid')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'passed\n')
