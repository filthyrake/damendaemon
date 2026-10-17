"""Object configuration is bounded and unambiguous before architecture dispatch."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from damendaemon import models

LIMIT = 1024 * 1024


class ModelConfigReaderTests(unittest.TestCase):
    def test_invalid_metadata_never_reaches_decoder_dispatch(self):
        cases = [b'[]', b'null', b'{"model_type":"llama","hidden_size":4,"hidden_size":8}',
                 b'{"model_type":"llama","nested":{"x":1,"x":2}}',
                 *[b'{"model_type":"llama","extra":'+value+b'}' for value in (b'NaN', b'Infinity', b'-Infinity', b'1e999')],
                 b'{"model_type":"llama","extra":"'+b'x'*LIMIT+b'"}',
                 b'{"model_type":"llama","extra":'+b'['*10000+b'0'+b']'*10000+b'}']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for index, payload in enumerate(cases):
                with self.subTest(case=index):
                    path.write_bytes(payload)
                    with patch('damendaemon.config.decoder_config', side_effect=lambda value:value) as dispatch:
                        with self.assertRaises(ValueError): models.config(directory)
                        dispatch.assert_not_called()

    def test_only_supported_mamba2_time_step_infinity_is_accepted(self):
        from damendaemon.config import read_config
        cases=[{'model_type':'llama','time_step_limit':[0,float('inf')]},
               {'model_type':'mamba2','extra':float('inf')},
               {'model_type':'mamba2','time_step_limit':[0,float('inf')],'extra':float('inf')},
               *[{'model_type':'mamba2','time_step_limit':limit} for limit in
                 ([0,float('nan')],[0,-float('inf')],[False,float('inf')],
                  [1,float('inf')],[float('inf'),0],[0,{'value':float('inf')}],
                  [0,float('inf'),1])]]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            for value in cases:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):read_config(path)
            for literal in ('1e999','-Infinity','NaN'):
                path.write_text('{"model_type":"mamba2","time_step_limit":[0,'+literal+']}')
                with self.assertRaises(ValueError):read_config(path)
            valid={'model_type':'mamba2','time_step_limit':[0.0,float('inf')]}
            path.write_text(json.dumps(valid))
            self.assertEqual(read_config(path),valid)

    def test_actual_read_is_bounded_without_trusting_file_stat(self):
        reads = []
        class Reader(io.BytesIO):
            def read(self, count=-1):
                reads.append(count)
                return super().read(count)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, 'open', return_value=Reader(b'{"model_type":"llama"}')), \
                    patch('damendaemon.config.decoder_config', side_effect=lambda value:value):
                self.assertEqual(models.config(directory), {'model_type':'llama'})
            self.assertEqual(reads, [LIMIT+1])

    def test_exact_limit_valid_object_preserves_values(self):
        value = {'model_type':'llama', 'extra':{'flag':False, 'missing':None, 'scale':1.5}}
        encoded = json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_bytes(encoded + b' '*(LIMIT-len(encoded)))
            with patch('damendaemon.config.decoder_config', side_effect=lambda value:value):
                self.assertEqual(models.config(directory), value)
