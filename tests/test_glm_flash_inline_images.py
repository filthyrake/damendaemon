import base64
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from dllm.glm_flash_inline_images import decode_inline_images
from dllm.glm_flash_images import CONTRACT
from dllm.glm_flash_vision import vision_specs,ROOT
from dllm.glm_text import EOS
from tests import test_glm_flash_image_generation,test_glm_flash_reference_loader,test_glm_flash_vision
from tests.test_packed_checkpoint import write_raw
from scripts.run_glm_flash_image_reference import main


def messages():
    buffer=io.BytesIO();Image.new('RGB',(112,112),(12,34,56)).save(buffer,format='PNG')
    return [dict(role='user',content=[dict(type='text',text='Describe.'),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()))])]


def snapshot(path):
    text=test_glm_flash_image_generation.text_fixture(path)
    config,_,_=test_glm_flash_reference_loader.fixture(path,vocab_size=text.vocab_size)
    c,_=test_glm_flash_vision.fixture();c.update(patch_size=14,out_hidden_size=config['text_config']['hidden_size'])
    config['vision_config']=c;config['quantization_config']['modules_to_not_convert'].append('model.visual')
    config['text_config'].update(eos_token_id=sorted(text.eos),pad_token_id=text.tokenizer.token_to_id(EOS),bos_token_id=None)
    config['image_token_id']=text.image_token_id
    (path/'config.json').write_text(json.dumps(config))
    (path/'processor_config.json').write_text(json.dumps(dict(processor_class='Glm5NextProcessor',image_processor=CONTRACT)))
    index=json.loads((path/'model.safetensors.index.json').read_text())
    filename='part-0.safetensors';raw=(path/filename).read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+size]);payload=bytearray(raw[8+size:])
    old='model.visual.unqualified';new='model.language_model.layers.3.visual_fixture'
    header[new]=header.pop(old);index['weight_map'][new]=index['weight_map'].pop(old)
    for name,shape in vision_specs(c).items():
        value=torch.ones(shape)*.02;data=bytes(value.view(torch.uint8).flatten().tolist());start=len(payload);payload.extend(data)
        header[ROOT+name]=dict(dtype='F32',shape=list(shape),data_offsets=[start,len(payload)]);index['weight_map'][ROOT+name]=filename
    write_raw(path/filename,header,payload)
    (path/'model.safetensors.index.json').write_text(json.dumps(index))
    # Controlled fixture head emits mapped token zero, not an image token.
    headfile=path/index['weight_map']['lm_head.weight'];raw=headfile.read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+size]);payload=bytearray(raw[8+size:]);start,end=header['lm_head.weight']['data_offsets'];payload[start:end]=bytes(end-start)
    write_raw(headfile,header,payload)


class FlashInlineImageTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def test_decoded_structure_pixels_identity_and_immutability(self):
        source=messages();original=deepcopy(source);result=decode_inline_images(source)
        self.assertEqual(result.messages,[dict(role='user',content=[dict(type='text',text='Describe.'),dict(type='image')])])
        self.assertEqual(result.images[0][0,0].tolist(),[12,34,56]);self.assertEqual(len(result.identities[0]),64)
        self.assertEqual(source,original)
        for options in (dict(max_bytes=1),dict(max_pixels=1)):
            with self.assertRaises(ValueError):decode_inline_images(source,**options)
        bad=messages();bad[0]['content'][1]['image_url']['url']='https://example.test/image.png'
        with self.assertRaises(ValueError):decode_inline_images(bad)
        with patch('PIL.Image.open',side_effect=AssertionError('decoded before template checks')):
            with self.assertRaises(ValueError):decode_inline_images([dict(role='assistant',content=source[0]['content'])])

    def test_actual_tiny_snapshot_inline_image_command(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);snapshot(path);message_file=path/'messages.json';message_file.write_text(json.dumps(messages()))
            output=io.StringIO()
            with redirect_stdout(output):code=main([str(path),'--messages',str(message_file),'--context','256','--max-new-tokens','2'])
            self.assertEqual(code,0);events=[json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[0]['modality'],'still_image');self.assertFalse(events[0]['official_model_qualified'])
            self.assertEqual(events[-1]['finish_reason'],'length');self.assertEqual(events[-1]['usage']['completion_tokens'],2)

    def test_command_bad_context_and_processor_reject_before_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);snapshot(path);message_file=path/'messages.json';message_file.write_text(json.dumps(messages()))
            with patch('scripts.run_glm_flash_image_reference.load_reference_model',side_effect=AssertionError('loaded weights')),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:main([str(path),'--messages',str(message_file),'--context','1'])
                self.assertEqual(error.exception.code,2)
                (path/'processor_config.json').write_text('{}')
                with self.assertRaises(SystemExit):main([str(path),'--messages',str(message_file)])
