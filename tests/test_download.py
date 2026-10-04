import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dllm.download import INDEX,snapshot_files
from dllm.cli import download


class DownloadTests(unittest.TestCase):
    def test_index_selects_only_referenced_shards_and_preserves_metadata(self):
        files=[INDEX,'config.json','tokenizer.json','chat_template.jinja','LICENSE','README.md',
            '1_Pooling/config.json','model.safetensors','part-a.safetensors','shards/part-b.safetensors',
            'original/model.safetensors','metal/model.safetensors']
        result=snapshot_files(files,{'weight_map':{'a':'part-a.safetensors','b':'shards/part-b.safetensors','c':'part-a.safetensors'}})
        self.assertEqual([n for n in result if n.endswith('.safetensors')],['part-a.safetensors','shards/part-b.safetensors'])
        for name in (INDEX,'LICENSE','README.md','chat_template.jinja','1_Pooling/config.json'):self.assertIn(name,result)

    def test_unsharded_uses_only_root_checkpoint(self):
        self.assertEqual(snapshot_files(['model.safetensors','original/model.safetensors','config.json']),['config.json','model.safetensors'])

    def test_missing_or_invalid_index_does_not_fall_back_to_other_weights(self):
        for index in (None,{},[],{'weight_map':{}},{'weight_map':[]},{'weight_map':{'a':None}}, {'weight_map':{'a':'missing.safetensors'}}):
            with self.subTest(index=index),self.assertRaises(ValueError):snapshot_files([INDEX,'model.safetensors'],index)
        with self.assertRaises(ValueError):snapshot_files(['original/model.safetensors'])

    def test_index_cannot_escape_or_expand_globs(self):
        for name in ('../model.safetensors','/model.safetensors','a/../model.safetensors',
                     './model.safetensors','a//model.safetensors','a\\model.safetensors',
                     '*.safetensors','model[1].safetensors','model?.safetensors','weights.bin'):
            with self.subTest(name=name),self.assertRaises(ValueError):snapshot_files([INDEX,name],{'weight_map':{'a':name}})

    def test_download_pins_index_and_snapshot_to_resolved_revision(self):
        files=[INDEX,'part.safetensors','original/model.safetensors','config.json']
        info=SimpleNamespace(sha='immutable-sha',siblings=[SimpleNamespace(rfilename=n) for n in files])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/INDEX;path.write_text(json.dumps({'weight_map':{'a':'part.safetensors'}}))
            with patch('huggingface_hub.HfApi') as api,patch('huggingface_hub.hf_hub_download',return_value=str(path)) as metadata,patch('huggingface_hub.snapshot_download',return_value='/snapshot') as snapshot:
                api.return_value.model_info.return_value=info
                result=download('owner/model','moving-branch')
                metadata.assert_called_once_with('owner/model',INDEX,revision='immutable-sha')
                snapshot.assert_called_once_with('owner/model',revision='immutable-sha',allow_patterns=sorted([INDEX,'part.safetensors','config.json']))
                self.assertEqual(result['revision'],'immutable-sha')

    def test_bad_index_stops_before_snapshot_transfer(self):
        info=SimpleNamespace(sha='sha',siblings=[SimpleNamespace(rfilename=INDEX)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/INDEX;path.write_text('{}')
            with patch('huggingface_hub.HfApi') as api,patch('huggingface_hub.hf_hub_download',return_value=str(path)),patch('huggingface_hub.snapshot_download') as snapshot:
                api.return_value.model_info.return_value=info
                with self.assertRaises(ValueError):download('owner/model')
                snapshot.assert_not_called()
