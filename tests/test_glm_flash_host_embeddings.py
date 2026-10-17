from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import torch
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.glm_flash_image_embeddings import image_embeddings
from tests import test_glm_flash_host_pipeline as host, test_host_tls


class FlashHostEmbeddingTests(unittest.TestCase):
    setUp=host.FlashHostPipelineTests.setUp
    session=host.FlashHostPipelineTests.session
    command=host.FlashHostPipelineTests.command
    pump=host.FlashHostPipelineTests.pump

    def plan(self,config=None,count=3,**options):
        return host.FlashHostPipelineTests.plan(self,config,count,**(dict(input_kind='flash_embeddings')|options))

    def test_actual_two_three_stage_features_chunks_reordered_decode_and_reuse(self):
        for count in (2,3):
            _,owners,control=self.session(count);first=owners[0]
            prepared=[torch.randn(4,16)*.2,torch.randn(3,16)*.3]
            states=[];expected=[]
            for values in prepared:
                logits,state=control.prefill_embeddings(values[None],capacity=16)
                states.append(state);self.addCleanup(state.close);expected.append(logits)
            original=owners[-1].stage.output_logits
            def compare(wanted):
                def project(hidden):
                    actual=original(hidden)
                    torch.testing.assert_close(actual,wanted,rtol=3e-5,atol=3e-6)
                    return actual
                return project
            # Unequal independent slots, a continued chunk, then reversed decode.
            self.command(owners,lambda:first.prefill_embeddings(prepared[0][:2],slot=1))
            with patch.object(owners[-1].stage,'output_logits',compare(expected[0])):
                self.command(owners,lambda:first.prefill_embeddings(prepared[0][2:],slot=1,start=2))
            with patch.object(owners[-1].stage,'output_logits',compare(expected[1])):
                self.command(owners,lambda:first.prefill_embeddings(prepared[1],slot=0))
            wanted=torch.cat([control.forward(torch.tensor([[4]]),states[1],position=3),control.forward(torch.tensor([[5]]),states[0],position=4)])
            with patch.object(owners[-1].stage,'output_logits',compare(wanted)):
                self.command(owners,lambda:first.decode([4,5],[0,1]))
            self.command(owners,lambda:first.release([1,0]))
            self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))
            # Text remains valid in this low-level mode and exclusions still hold.
            self.command(owners,lambda:first.prefill_chunk([1,2],slot=1))
            self.command(owners,lambda:first.release([1]))
            with patch.object(owners[-1].stage,'output_logits',return_value=torch.arange(32,dtype=torch.float32)[None]):
                token=self.command(owners,lambda:first.prefill_embeddings(prepared[0],slot=0))
            self.assertEqual(token,28)
            self.command(owners,first.stop);self.assertTrue(all(owner.closed for owner in owners))

    def test_real_vision_projection_enters_language_stages_once(self):
        from damendaemon.glm_flash_vision import FlashVision,vision_specs
        from tests.test_glm_flash_vision import fixture
        c,_=fixture();c['out_hidden_size']=16
        torch.manual_seed(97);weights={name:torch.randn(shape)*.05 for name,shape in vision_specs(c).items()}
        vision=FlashVision(c,weights)
        features=vision.forward(torch.randn(8,24),[(1,2,4)])
        _,owners,control=self.session(3)
        values=image_embeddings(control,torch.tensor([[1,29,29,2]]),[features],image_token_id=29)[0]
        wanted,state=control.prefill_embeddings(values[None],capacity=16);self.addCleanup(state.close)
        original=owners[-1].stage.output_logits
        def project(hidden):
            actual=original(hidden);torch.testing.assert_close(actual,wanted,rtol=3e-5,atol=3e-6);return actual
        with patch.object(owners[0].stage,'embedding',side_effect=AssertionError('features embedded twice')),patch.object(owners[-1].stage,'output_logits',project):
            self.command(owners,lambda:owners[0].prefill_embeddings(values))
        self.command(owners,owners[0].stop)

    def test_malformed_input_and_cursor_reject_before_participant_admission(self):
        _,owners,_=self.session();first=owners[0];values=torch.zeros(2,16)
        bad=[None,values.double(),values[:,None],values[:,:15],values[:0],torch.zeros(9,16),torch.full((2,16),float('nan')),torch.empty(2,16,device='meta')]
        with patch.object(first.channels[0],'begin',side_effect=AssertionError('participant admitted invalid input')):
            for value in bad:
                with self.subTest(value=type(value)):
                    with self.assertRaises(ValueError):first.prefill_embeddings(value)
            for options in (dict(slot=True),dict(slot=2),dict(start=True),dict(start=1),dict(start=16)):
                with self.assertRaises(ValueError):first.prefill_embeddings(values,**options)
            with patch.object(first.stage.stage,'max_output_bytes',127):
                with self.assertRaises(ValueError):first.prefill_embeddings(values)
        self.assertTrue(first.ready);self.assertTrue(all(owner.channel.state.sequence==1 for owner in owners))
        self.command(owners,lambda:first.prefill_embeddings(values))
        with patch.object(first.channels[0],'begin',side_effect=AssertionError('bad cursor admitted')):
            for start in (0,1,3,16):
                with self.assertRaises(ValueError):first.prefill_embeddings(values,start=start)
        self.command(owners,first.stop)

    def test_input_contract_identity_and_default_family_admission(self):
        plans=[self.plan(),self.plan(input_kind='tokens')]
        self.assertNotEqual(plans[0].identity,plans[1].identity)
        for value in ('images',None,True):
            with self.assertRaises(ValueError):self.plan(input_kind=value)
        from damendaemon.host_topology import HostTopology
        from tests.test_glm_checkpoint import config
        with self.assertRaises(ValueError):HostTopology(config(),[(0,2),(2,3)],b'g'*32,quantization='fp8_reference',input_kind='flash_embeddings')
        tls=test_host_tls.HostTlsTests();self.addCleanup(tls.doCleanups);links=[]
        for rank,sock in enumerate(tls.pair()):
            link=HostTensorChannel(sock,plans[rank].stage_identity(1),rank,max_frame_bytes=4096,timeout=2)
            self.addCleanup(link.close);links.append(link)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(link.handshake) for link in links]
            for future in futures:
                with self.assertRaises((ValueError,RuntimeError,EOFError,OSError)):future.result(5)
        with patch.object(self,'plan',side_effect=lambda **kw:host.FlashHostPipelineTests.plan(self,**kw)):
            _,owners,_=self.session()
        with self.assertRaises(ValueError):owners[0].prefill_embeddings(torch.zeros(2,16))
        self.assertTrue(owners[0].ready);self.command(owners,owners[0].stop)

    def test_partial_compute_and_retained_cleanup_retire_epoch(self):
        _,owners,_=self.session(3)
        with patch.object(owners[1].stage,'forward_layers',side_effect=RuntimeError('middle failure')),patch.object(owners[0].stage,'close',side_effect=RuntimeError('retry cleanup')):
            self.command(owners,lambda:owners[0].prefill_embeddings(torch.zeros(2,16)),fails=True)
            self.assertTrue(owners[0].failed);self.assertFalse(owners[0].closed)
            self.assertTrue(all(owner.failed for owner in owners))
        owners[0].close();self.assertTrue(owners[0].closed)
        self.assertTrue(all(owner.stage.states==[None,None] for owner in owners))
        with self.assertRaises(RuntimeError):owners[0].prefill_embeddings(torch.zeros(2,16))

    def test_peer_loss_during_prepared_prefill_retires_epoch(self):
        _,owners,_=self.session();owners[1].close()
        with self.assertRaises((RuntimeError,ValueError,EOFError,OSError)):
            owners[0].prefill_embeddings(torch.zeros(2,16))
        self.assertTrue(owners[0].failed);self.assertTrue(owners[0].stage.closed)

    def test_native_text_owner_rejects_prepared_feature_mode(self):
        from tests.test_glm_flash_host_generation import FlashHostGenerationTests
        from damendaemon.glm_flash_host_generation import HostFlashGenerator
        fixture=FlashHostGenerationTests();self.addCleanup(fixture.doCleanups)
        text,workers,_=fixture.session(input_kind='flash_embeddings')
        with self.assertRaisesRegex(ValueError,'native host generation'):
            HostFlashGenerator(workers[0],text,chunk_size=8)

    def test_input_storage_is_detached_before_peer_admission(self):
        _,owners,control=self.session();first=owners[0];values=torch.randn(2,16)
        wanted,state=control.prefill_embeddings(values[None].clone(),capacity=16);self.addCleanup(state.close)
        begin=first.channels[0].begin;project=owners[-1].stage.output_logits
        def admit(command):
            values.fill_(0);return begin(command)
        def compare(hidden):
            result=project(hidden);torch.testing.assert_close(result,wanted,rtol=3e-5,atol=3e-6);return result
        with patch.object(first.channels[0],'begin',admit),patch.object(owners[-1].stage,'output_logits',compare):
            self.command(owners,lambda:first.prefill_embeddings(values))
        self.command(owners,first.stop)
