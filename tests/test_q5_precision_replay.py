import unittest
import numpy as np
from scripts.experiments.q5_precision_replay import CHECKPOINT_SHA256, metrics, teacher_case


class PrecisionReplayTests(unittest.TestCase):
    def test_gate_retains_failure_and_teacher_tokens(self):
        reference=np.array([[1.,2.,3.],[2.,3.,4.]],dtype=np.float32)
        exact=metrics(reference,reference,[2,2])
        self.assertTrue(exact['passed'])
        changed=reference.copy();changed[1,0]+=.2
        result=metrics(changed,reference,[2,2])
        self.assertFalse(result['passed'])
        self.assertEqual(result['errors'][1]['teacher_token'],2)
        self.assertEqual(result['errors'][1]['candidate_token'],2)

    def test_nonfinite_and_wrong_shape_rejected(self):
        for candidate in (np.array([[float('nan')]]),np.array([[float('inf')]]),np.ones((2,1))):
            with self.assertRaises(ValueError):metrics(candidate,np.ones((1,1)),[0])

    def test_teacher_pinned_and_fixed(self):
        case=dict(case=0,graph=False,prompt_tokens=48,reference_tokens=list(range(8)))
        report=dict(sha256=CHECKPOINT_SHA256,cases=[case])
        self.assertEqual(teacher_case(report),list(range(8)))
        for key,value in [('reference_tokens',[1]),('reference_tokens',[True]*8),('prompt_tokens',47)]:
            bad=dict(case);bad[key]=value
            with self.assertRaises(ValueError):teacher_case(dict(report,cases=[bad]))
        with self.assertRaises(ValueError):teacher_case(dict(report,sha256='wrong'))


if __name__=='__main__':unittest.main()
