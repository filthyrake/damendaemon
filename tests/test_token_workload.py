import unittest
from damendaemon.token_workload import token_workload_digest


class TokenWorkloadTests(unittest.TestCase):
    def test_identity_preserves_tokens_prompt_boundaries_order_and_output_budget(self):
        baseline=token_workload_digest([[1,2],[3]],4)
        self.assertEqual(baseline,token_workload_digest([[1,2],[3]],4))
        for prompts,limit in [([[1,9],[3]],4),([[1],[2,3]],4),([[3],[1,2]],4),([[1,2],[3]],5)]:
            with self.subTest(prompts=prompts,limit=limit):
                self.assertNotEqual(baseline,token_workload_digest(prompts,limit))
        self.assertEqual(len(baseline),64)

    def test_invalid_or_ambiguous_token_workloads_reject(self):
        for prompts in (None,[],[[]],[[True]],[[1.0]],[[-1]],['1'],[[[1]]]):
            with self.subTest(prompts=prompts),self.assertRaises(ValueError):token_workload_digest(prompts,1)
        for limit in (None,True,1.0,0,-1):
            with self.subTest(limit=limit),self.assertRaises(ValueError):token_workload_digest([[1]],limit)

    def test_teacher_lookup_preserves_diagnostic_inputs_without_aliasing(self):
        from damendaemon.token_workload import teacher_token_lookup
        prompts=[[1]]
        teacher=dict(passed=False,workload_sha256=token_workload_digest(prompts,2),cases=[
            dict(graph=graph,case=0,prompt_tokens=1,reference_tokens=[2,3]) for graph in (False,True)])
        lookup=teacher_token_lookup(teacher,prompts,2,4)
        self.assertEqual(lookup,{(False,0):[2,3],(True,0):[2,3]})
        teacher['cases'][0]['reference_tokens'][0]=0
        self.assertEqual(lookup[(False,0)],[2,3])
        self.assertEqual(teacher_token_lookup(None,prompts,2,4),{})
