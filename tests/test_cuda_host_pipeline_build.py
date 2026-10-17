import unittest
from unittest.mock import patch
from damendaemon.cuda_host_pipeline_build import build_cuda_host_pipeline_worker,CudaHostPipelineBuildError
from tests import test_glm_host_pipeline_build as glm_tests


class CudaHostBuildTests(unittest.TestCase):
    setUp=glm_tests.GlmHostPipelineBuildTests.setUp
    plan=glm_tests.GlmHostPipelineBuildTests.plan
    links=glm_tests.GlmHostPipelineBuildTests.links

    def build(self,plan,links,**options):
        controls=dict(device='cuda:0',max_device_bytes=1024**2,max_transfer_bytes=4096,epoch=1)|options
        return build_cuda_host_pipeline_worker(self.path,plan,0,links,**controls)

    def test_invalid_controls_do_not_claim_links(self):
        plan=self.plan();links=self.links(plan)[0]
        for options in (dict(device='cpu'),dict(device='cuda'),dict(max_device_bytes=0),dict(max_transfer_bytes=True),
                        dict(device_workspace_bytes=1),dict(max_cache_bytes=0),dict(verify_source=1)):
            with self.assertRaises(ValueError):self.build(plan,links,**options)
            self.assertTrue(all(not t.closed and getattr(t,'_cpu_worker_builder_owner',None) is None for t in links))

    def test_placement_failure_closes_claimed_links(self):
        plan=self.plan();links=self.links(plan)[0]
        with patch('damendaemon.cuda_host_pipeline_build.place_cuda_stage',side_effect=ValueError('placement admission')):
            with self.assertRaises(CudaHostPipelineBuildError) as caught:self.build(plan,links)
        self.assertTrue(caught.exception.closed);self.assertTrue(all(t.closed for t in links))

    def test_device_payload_rejection_precedes_copy(self):
        plan=self.plan();links=self.links(plan)[0]
        with patch('damendaemon.cuda_stage_placement._copy',side_effect=AssertionError('premature copy')):
            with self.assertRaises(CudaHostPipelineBuildError) as caught:self.build(plan,links,max_device_bytes=1)
        self.assertIn('raw payload',str(caught.exception));self.assertTrue(caught.exception.closed)

    def test_link_cleanup_failure_keeps_retry_owner(self):
        plan=self.plan();links=self.links(plan)[0]
        with patch.object(links[0],'close',side_effect=OSError('close')):
            with self.assertRaises(CudaHostPipelineBuildError) as caught:self.build(plan,links,max_device_bytes=1)
        self.assertFalse(caught.exception.closed);self.assertTrue(links[1].closed)
        caught.exception.close();self.assertTrue(caught.exception.closed);self.assertTrue(links[0].closed)
