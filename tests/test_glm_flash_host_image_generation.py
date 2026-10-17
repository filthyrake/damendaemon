from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_host_image_generation import HostFlashImageGenerator,native_image_session_identity,load_host_image_vision
from damendaemon.glm_flash_host_generation import HostFlashGenerator,native_session_identity,native_excluded_ids
from damendaemon.glm_flash_image_tool_text import FlashImageToolText
from damendaemon.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator
from damendaemon.glm_flash_host_pipeline_build import build_glm_flash_host_pipeline_worker
from damendaemon.glm_flash_reference_loader import load_reference_model
from damendaemon.glm_flash_vision_loader import load_reference_vision
from damendaemon.glm_flash_images import prepare_images
from damendaemon.glm_flash_image_embeddings import image_embeddings
from damendaemon.host_topology import HostTopology
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.cpu_snapshot_identity import fingerprint
from damendaemon.glm_flash_tools import FlashTools
from damendaemon.glm_text import END_THINK,EOS
from damendaemon.stage_protocol import PREFILL_CHUNK
from damendaemon.reference_cancellation import cancellation_scope,ReferenceCancelled
from tests import test_host_tls,test_host_pipeline_worker
from tests.test_glm_flash_inline_images import snapshot
from tests.test_glm_tools import TOOLS,CALLS


class FlashHostImageGenerationTests(unittest.TestCase):
    pump=test_host_pipeline_worker.HostPipelineWorkerTests.pump

    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name);snapshot(self.path)
        self.config=json.loads((self.path/'config.json').read_text())
        self.text=FlashImageToolText(self.path,vocab_size=self.config['text_config']['vocab_size'])
        self.messages=[dict(role='user',content=[dict(type='text',text='Hi'),dict(type='image')])]
        self.images=[torch.full((112,112,3),127,dtype=torch.uint8)]

    def session(self,count=2,verified=False,claim=True,**limits):
        source=bytes.fromhex(fingerprint(self.path).model_id) if verified else b'i'*32
        plan=HostTopology(self.config,[(0,1),(1,2),(2,3)] if count==3 else [(0,2),(2,3)],source,
            quantization='fp8_reference',context=4096,context_limit=4096,slots=2,chunk_size=256,frame_bytes=131072,
            mapped_vocab_size=self.text.tokenizer_size,excluded_ids=native_excluded_ids(self.text),input_kind='flash_embeddings',
            session_identity=native_image_session_identity(self.text,verified=verified))
        groups=[[] for _ in range(count)]
        for index in range(1,count):
            tls=test_host_tls.HostTlsTests();self.addCleanup(tls.doCleanups)
            for rank,sock in enumerate(tls.pair()):
                transport=HostTensorChannel(sock,plan.stage_identity(index),rank,max_frame_bytes=plan.frame_bytes,timeout=3)
                self.addCleanup(transport.close);groups[0 if rank==0 else index].append(transport)
        owners=[build_glm_flash_host_pipeline_worker(self.path,plan,index,group,epoch=9,verify_source=verified) for index,group in enumerate(groups)]
        for owner in owners:self.addCleanup(owner.close)
        with ThreadPoolExecutor(count) as pool:
            futures=[pool.submit(owner.handshake) for owner in owners]
            for future in futures:future.result(5)
        errors,threads=self.pump(owners)
        self.addCleanup(lambda:[thread.join(3) for thread in threads]);self.addCleanup(owners[0].close)
        vision=load_host_image_vision(self.path,owners[0])
        runner=HostFlashImageGenerator(owners[0],self.text,vision,chunk_size=32,**limits) if claim else None
        if runner:self.addCleanup(runner.close)
        return runner,owners,vision

    def control(self):
        return FlashImageToolReferenceGenerator(load_reference_model(self.path),self.text,load_reference_vision(self.path),max_context=4096)

    def test_two_three_stage_real_images_match_full_model_native_and_hidden_controls(self):
        for count in (2,3):
            runner,owners,_=self.session(count);control=self.control()
            for options in ({},dict(reasoning_effort='high',clear_thinking=True)):
                expected=list(control.generate_images(self.messages,self.images,max_new_tokens=2,**options))
                actual=list(runner.generate_images(self.messages,self.images,max_new_tokens=2,**options))
                self.assertEqual(actual,expected);self.assertTrue(all(not any(owner.channel.state.lengths) for owner in owners))
            # Nonzero full-stream hidden comparison avoids relying on a zero fixture head.
            patches,grids=prepare_images(self.images);features=control.vision.forward(patches,grids)
            tokens=self.text.image_chat_tokens(self.messages,grids)
            values=image_embeddings(control.model,torch.tensor([tokens]),[features],image_token_id=self.text.image_token_id)
            captured=[];original=control.model._logits
            def capture(hidden,all_logits):captured.append(hidden[:,-1].clone());return original(hidden,all_logits)
            with patch.object(control.model,'_logits',capture):_,state=control.model.prefill_embeddings(values,capacity=4096)
            state.close();actual_hidden=[];project=owners[-1].stage.output_logits
            def record(hidden):actual_hidden.append(hidden.clone());return project(hidden)
            with patch.object(owners[-1].stage,'output_logits',record):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
            torch.testing.assert_close(actual_hidden[-1],captured[0],rtol=3e-5,atol=3e-6)
            self.assertGreater(float(captured[0].abs().max()),0)
            self.assertEqual(list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=1))[-1]['finish_reason'],'length')
            runner.close()

    def scripted(self,runner,owners,wire):
        tokens=iter(self.text.tokenizer.encode(wire,add_special_tokens=False).ids);project=owners[-1].stage.output_logits
        def logits(hidden):
            result=project(hidden);command=owners[-1].channel.state.pending
            # Ignore continuation samples of incomplete prefill chunks.
            if command.operation!=PREFILL_CHUNK or command.entries[0][2]+command.rows>=self.prompt_length:
                token=next(tokens);result.fill_(-100);result[:,token]=100
                result[:,self.text.image_token_id]=1000
            return result
        return patch.object(owners[-1].stage,'output_logits',logits)

    def test_native_image_tools_parse_and_reuse_actual_stage_execution(self):
        runner,owners,_=self.session();wire='reason'+END_THINK+FlashTools(TOOLS).render(CALLS)+EOS
        self.prompt_length=len(self.text.image_tool_chat_tokens(self.messages,[[1,8,8]],TOOLS))
        with self.scripted(runner,owners,wire):
            events=list(runner.generate_image_tools(self.messages,self.images,TOOLS,max_new_tokens=512))
        self.assertEqual(events[-1]['finish_reason'],'tool_calls')
        calls=next(event['tool_calls'] for event in events if 'tool_calls' in event)
        self.assertEqual(json.loads(calls[0]['function']['arguments']),json.loads(CALLS[0]['function']['arguments']))
        self.assertEqual(events[-1]['usage']['prompt_tokens'],self.prompt_length)
        self.assertTrue(all(not any(owner.channel.state.lengths) for owner in owners))
        history=self.messages+[dict(role='assistant',content='',tool_calls=calls),dict(role='tool',tool_call_id=calls[0]['id'],content='ok')]
        self.assertEqual(list(runner.generate_image_tools(history,self.images,TOOLS,max_new_tokens=1))[-1]['finish_reason'],'length')

    def test_invalid_requests_reject_before_vision_and_dispatch(self):
        runner,owners,_=self.session();before=owners[0].channel.state.sequence
        with patch.object(runner.vision,'forward',side_effect=AssertionError('vision before validation')),patch.object(owners[0],'prefill_embeddings',side_effect=AssertionError('host dispatch')):
            for messages,images,tools,limit in ((self.messages,[],TOOLS,2),(self.messages,self.images,[],2),
                    (self.messages,self.images,TOOLS,4096),(self.messages,self.images,TOOLS,True),
                    ([dict(role='user',content=[dict(type='video')])],self.images,TOOLS,2),
                    (self.messages,[self.images[0].float()],TOOLS,2)):
                with self.assertRaises(ValueError):list(runner.generate_image_tools(messages,images,tools,max_new_tokens=limit))
            with patch.object(runner,'image_output_bytes',1):
                with self.assertRaises(ValueError):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
            with patch.object(runner,'feature_output_bytes',1):
                with self.assertRaises(ValueError):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
        self.assertEqual(owners[0].channel.state.sequence,before);self.assertTrue(owners[0].ready)

    def test_policy_geometry_and_verified_vision_admission(self):
        from damendaemon.glm_flash_tool_text import FlashToolText
        text=FlashToolText(self.path,vocab_size=self.text.vocab_size)
        self.assertNotEqual(native_image_session_identity(self.text),native_session_identity(text))
        self.assertNotEqual(native_image_session_identity(self.text),native_image_session_identity(self.text,verified=True))
        _,owners,vision=self.session(verified=True,claim=False)
        with self.assertRaisesRegex(ValueError,'verified native images'):
            HostFlashImageGenerator(owners[0],self.text,load_reference_vision(self.path),chunk_size=32)
        self.assertIsNone(getattr(owners[0],'_completion_owner',None))
        with self.assertRaises(ValueError):HostFlashGenerator(owners[0],text,chunk_size=32)
        bad=deepcopy(self.text);bad.image_token_id+=1
        with self.assertRaises(ValueError):native_image_session_identity(bad)
        with patch.object(vision,'c',dict(vision.c,patch_size=2)):
            with self.assertRaises(ValueError):HostFlashImageGenerator(owners[0],self.text,vision,chunk_size=32)
        runner=HostFlashImageGenerator(owners[0],self.text,vision,chunk_size=32);self.addCleanup(runner.close)
        self.assertEqual(list(runner.generate_images(self.messages,self.images,max_new_tokens=1))[-1]['finish_reason'],'length')
        runner.close();self.assertIsNone(runner.vision)

    def test_vision_failure_and_cancellation_before_dispatch_preserve_session(self):
        runner,owners,_=self.session();before=owners[0].channel.state.sequence
        with patch.object(runner.vision,'forward',side_effect=RuntimeError('vision failure')):
            with self.assertRaisesRegex(RuntimeError,'vision failure'):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
        self.assertTrue(owners[0].ready);self.assertEqual(owners[0].channel.state.sequence,before)
        for bad in (None,torch.zeros(1,1),torch.full((16,16),float('nan'))):
            with patch.object(runner.vision,'forward',return_value=bad):
                with self.assertRaises(ValueError):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
        self.assertTrue(owners[0].ready);self.assertEqual(owners[0].channel.state.sequence,before)
        with cancellation_scope(lambda:True):
            with self.assertRaises(ReferenceCancelled):list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
        self.assertTrue(owners[0].ready)
        self.assertEqual(list(runner.generate_images(self.messages,self.images,max_new_tokens=1))[-1]['finish_reason'],'length')

    def test_abandonment_parser_failure_and_cleanup_retry_retire_epoch(self):
        runner,owners,_=self.session();stream=runner.generate_images(self.messages,self.images,max_new_tokens=3)
        next(stream);stream.close();self.assertTrue(runner.completion.stopped.is_set())
        runner,owners,_=self.session();self.prompt_length=len(self.text.image_tool_chat_tokens(self.messages,[[1,8,8]],TOOLS))
        with self.scripted(runner,owners,'reason'+END_THINK+'<tool_call>not json</tool_call>'+EOS):
            with self.assertRaises(ValueError):list(runner.generate_image_tools(self.messages,self.images,TOOLS,max_new_tokens=128))
        self.assertTrue(runner.completion.stopped.is_set())
        runner,owners,_=self.session();stream=runner.generate_images(self.messages,self.images,max_new_tokens=3);next(stream)
        with patch.object(owners[0].stage,'close',side_effect=RuntimeError('retry cleanup')):
            stream.close()
        self.assertFalse(owners[0].closed);runner.close();self.assertTrue(owners[0].closed);self.assertIsNone(runner.vision)

    def test_request_pixels_messages_and_tokenizer_are_detached(self):
        runner,owners,_=self.session()
        patches,grids=prepare_images(self.images);wanted=patches.clone()
        prompt_length=len(self.text.image_chat_tokens(self.messages,grids))
        original_stream=runner._stream;original_vision=runner.vision.forward
        def stream(codec):
            self.images[0].zero_();self.messages.clear()
            self.text.tokenizer.add_tokens(['detached-request-token'])
            return original_stream(codec)
        def vision(patches,grids):
            torch.testing.assert_close(patches,wanted,rtol=0,atol=0)
            return original_vision(patches,grids)
        with patch.object(runner,'_stream',stream),patch.object(runner.vision,'forward',vision):
            events=list(runner.generate_images(self.messages,self.images,max_new_tokens=1))
        self.assertEqual(events[-1]['usage']['prompt_tokens'],prompt_length)
        self.assertNotEqual(self.text.tokenizer.get_vocab_size(),runner.text.tokenizer.get_vocab_size())

    def test_verified_vision_detects_source_mismatch_and_change_during_load(self):
        _,owners,_=self.session(verified=True,claim=False)
        path=self.path/'tokenizer.json';original=path.read_bytes();path.write_bytes(original+b' ')
        with patch('damendaemon.glm_flash_vision_loader.load_reference_vision',side_effect=AssertionError('payload before source validation')):
            with self.assertRaisesRegex(ValueError,'snapshot differs'):load_host_image_vision(self.path,owners[0])
        path.write_bytes(original)
        def changed(*args,**kwargs):
            result=load_reference_vision(*args,**kwargs);path.write_bytes(original+b' ');return result
        with patch('damendaemon.glm_flash_vision_loader.load_reference_vision',changed):
            with self.assertRaisesRegex(ValueError,'snapshot file changed'):load_host_image_vision(self.path,owners[0])
        self.assertTrue(owners[0].ready);self.assertIsNone(getattr(owners[0],'_completion_owner',None))
