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
            with patch('dllm.glm_reference_loader.load_reference_model',side_effect=AssertionError('payload loaded')) as load:
                for options in (['--device','cuda'],['--device','cpu'],['--device','cuda:0','--device-weight-mib','512']):
                    with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),*options]))
                (path/'config.json').write_text(json.dumps(config|dict(max_position_embeddings=1)))
                with self.assertRaises(ValueError):prepare(parser.parse_args([str(path),'--device','cuda:0']))
                load.assert_not_called()

    def test_command_defaults_keep_serial_loopback_and_bounded_resources(self):
        args=parser_for_command().parse_args(['/local/snapshot','--device','cuda:1'])
        self.assertEqual(args.cpu_threads,1);self.assertEqual(args.queue_size,0)
        self.assertEqual(args.context,128);self.assertLess(args.device_weight_mib,args.allocator_mib)
        self.assertFalse(hasattr(args,'host'))


if __name__=='__main__':unittest.main()
