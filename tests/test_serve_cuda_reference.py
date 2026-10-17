import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.serve_cuda_reference import parser_for_command,prepare
from tests.test_glm_generation import text_fixture
from tests.test_glm_reference_loader import fixture


class ServeCudaTests(unittest.TestCase):
    def test_explicit_device_allocator_and_metadata_before_weight_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);config,_,_=fixture(path,vocab_size=text.vocab_size)
            parser=parser_for_command()
            with patch('damendaemon.glm_reference_loader.load_reference_model',side_effect=AssertionError('payload loaded')) as load:
                for options in (['--device','cuda'],['--device','cpu'],['--device','cuda:0','--device-weight-mib','512']):
                    with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),*options]))
                (path/'config.json').write_text(json.dumps(config|dict(max_position_embeddings=1)))
                with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),'--device','cuda:0']))
                load.assert_not_called()

    def test_flash_nested_context_rejected_before_payload(self):
        from tests.test_glm_flash_generation import text_fixture as flash_text,snapshot
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=flash_text(path);snapshot(path,text)
            config=json.loads((path/'config.json').read_text())
            config['text_config']['max_position_embeddings']=1
            (path/'config.json').write_text(json.dumps(config))
            with patch('damendaemon.glm_flash_reference_loader.load_reference_model',side_effect=AssertionError('payload loaded')) as load:
                with self.assertRaises(ValueError):prepare(parser_for_command().parse_args([str(path),'--device','cuda:0']))
                load.assert_not_called()

    def test_command_defaults_keep_serial_loopback_and_bounded_resources(self):
        args=parser_for_command().parse_args(['/local/snapshot','--device','cuda:1'])
        self.assertEqual(args.cpu_threads,1);self.assertEqual(args.queue_size,0)
        self.assertEqual(args.context,128);self.assertLess(args.device_weight_mib,args.allocator_mib)
        self.assertFalse(hasattr(args,'host'))


if __name__=='__main__':unittest.main()

class DeepSeekCudaCommandTests(unittest.TestCase):
    def test_context_metadata_fails_before_payload(self):
        from tests.test_deepseek_generation import text_fixture as deepseek_text
        from tests.test_deepseek_reference_loader import fixture as deepseek_fixture
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=deepseek_text(path);config,_=deepseek_fixture(path,vocab_size=text.vocab_size)
            (path/'config.json').write_text(json.dumps(config|dict(max_position_embeddings=1)))
            with patch('damendaemon.deepseek_reference_loader.load_reference_model',side_effect=AssertionError('payload loaded')):
                with self.assertRaises(ValueError):prepare(parser_for_command().parse_args([str(path),'--device','cuda:0']))

class FlashImageCudaCommandTests(unittest.TestCase):
    def test_explicit_image_optin_and_processor_before_payload(self):
        from tests.test_glm_flash_inline_images import snapshot
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);snapshot(path)
            parser=parser_for_command()
            with patch('damendaemon.glm_flash_reference_loader.load_reference_model',side_effect=AssertionError('payload loaded')):
                with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),'--device','cuda:0','--max-patches','64']))
                (path/'processor_config.json').write_text('{}')
                with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),'--device','cuda:0','--flash-images']))
