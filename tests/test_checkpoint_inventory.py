import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from dllm.qwen2 import weight_files


class CheckpointInventoryTests(unittest.TestCase):
    def test_duplicate_tensor_and_mismatched_index_reject_before_loading(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            save_file({'a':torch.ones(2)}, str(path/'one.safetensors'))
            save_file({'a':torch.zeros(2)}, str(path/'two.safetensors'))
            index = path/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'a':'one.safetensors', 'b':'two.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'duplicate tensor'):
                weight_files(path)
            index.write_text(json.dumps({'weight_map':{'b':'one.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'shard index'):
                weight_files(path)

    def test_nested_packed_shards_and_cache_symlinks_are_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'weights').mkdir()
            save_file({'a':torch.ones(2, dtype=torch.uint8)}, str(path/'blob'))
            (path/'weights'/'one.safetensors').symlink_to(path/'blob')
            (path/'model.safetensors.index.json').write_text(json.dumps({
                'weight_map':{'a':'weights/one.safetensors'}}))
            self.assertEqual(weight_files(path), [path/'weights'/'one.safetensors'])

    def test_malformed_index_and_header_fail_as_value_errors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            index = path/'model.safetensors.index.json'
            for content in ('{}', '[]', '{"weight_map":{}}',
                            '{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}',
                            '{"weight_map":{"a":"../escape.safetensors"}}',
                            '{"weight_map":{"a":"missing.safetensors"}}'):
                index.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    weight_files(path)
            index.unlink()
            (path/'model.safetensors').write_bytes(b'broken header')
            with self.assertRaisesRegex(ValueError, 'safetensors header'):
                weight_files(path)

    def test_shared_inventory_never_reads_tensor_values(self):
        from dllm.checkpoint import inspect_snapshot
        with TemporaryDirectory() as directory:
            path = Path(directory)
            save_file({'a':torch.ones(2, 3)}, str(path/'model.safetensors'))
            import dllm.checkpoint as checkpoint
            original = checkpoint.safe_open
            class HeadersOnly:
                def __init__(self, *args, **kwargs):
                    self.reader = original(*args, **kwargs)
                def __enter__(self):
                    self.reader.__enter__()
                    return self
                def __exit__(self, *args):
                    return self.reader.__exit__(*args)
                def keys(self):
                    return self.reader.keys()
                def get_slice(self, name):
                    return self.reader.get_slice(name)
                def get_tensor(self, name):
                    raise AssertionError('tensor values read')
            with patch.object(checkpoint, 'safe_open', HeadersOnly), \
                    patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU initialized')):
                files, metadata = inspect_snapshot(path)
            self.assertEqual(files, [path/'model.safetensors'])
            self.assertEqual(metadata['model.safetensors']['tensors']['a'], {'shape':[2, 3], 'dtype':'F32'})
