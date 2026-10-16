"""Malformed checkpoint metadata is rejected without tensor dependencies."""
import struct
from pathlib import Path
import tempfile
import unittest
from dllm.download import read_index
from dllm.safetensor_header import read_header


class CheckpointJsonTests(unittest.TestCase):
    def test_nested_index_is_a_validation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json';path.write_text('['*10000+'0'+']'*10000)
            with self.assertRaises(ValueError):read_index(path)

    def test_nested_safetensors_header_is_a_validation_failure(self):
        raw=('['*10000+'0'+']'*10000).encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors';path.write_bytes(struct.pack('<Q',len(raw))+raw)
            with path.open('rb') as reader,self.assertRaises(ValueError):read_header(reader)

    def test_shared_parser_preserves_values_and_rejects_duplicates(self):
        from dllm.safetensor_header import parse_metadata_json
        self.assertEqual(parse_metadata_json(b'{"shape":[1,2],"missing":null}'),{'shape':[1,2],'missing':None})
        self.assertEqual(parse_metadata_json('[1,2]'),[1,2])
        for raw in ('{"x":1,"x":2}','{"nested":{"x":1,"x":2}}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse_metadata_json(raw)
