import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dllm.download import INDEX,snapshot_files,gguf_file
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

    def test_duplicate_and_oversized_indexes_stop_before_shard_download(self):
        info = SimpleNamespace(sha='sha', siblings=[SimpleNamespace(rfilename=name)
            for name in (INDEX, 'part.safetensors')])
        valid = '{"weight_map":{"a":"part.safetensors"}}'
        invalid = (
            '{"weight_map":{},"weight_map":{"a":"part.safetensors"}}',
            '{"weight_map":{"a":"part.safetensors","a":"part.safetensors"}}',
            valid + ' ' * (16 * 1024 * 1024),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/INDEX
            for content in invalid:
                with self.subTest(size=len(content)), patch('huggingface_hub.HfApi') as api, \
                        patch('huggingface_hub.hf_hub_download', return_value=str(path)), \
                        patch('huggingface_hub.snapshot_download') as snapshot:
                    path.write_text(content)
                    api.return_value.model_info.return_value = info
                    with self.assertRaises(ValueError):
                        download('owner/model')
                    snapshot.assert_not_called()


class GgufDownloadTests(unittest.TestCase):
    def test_pins_and_fetches_only_the_selected_gguf(self):
        name = 'quantized/model.Q8_0.gguf'
        info = SimpleNamespace(sha='pinned-sha', siblings=[SimpleNamespace(rfilename=n)
            for n in (name, 'model.Q4_K_M.gguf', INDEX, 'part.safetensors')])
        with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download', return_value='/cache/snapshot/model.Q8_0.gguf') as fetch, patch('huggingface_hub.snapshot_download') as snapshot:
            api.return_value.model_info.return_value = info
            result = download('owner/model', 'moving-revision', name)
            api.return_value.model_info.assert_called_once_with('owner/model', revision='moving-revision')
            fetch.assert_called_once_with('owner/model', name, revision='pinned-sha')
            snapshot.assert_not_called()
            self.assertEqual(result['revision'], 'pinned-sha')
            self.assertEqual(result['filename'], name)

    def test_gguf_selection_rejects_globs_escapes_and_missing_files(self):
        for name in ('../a.gguf', '/a.gguf', './a.gguf', 'a//b.gguf', 'a/../b.gguf',
                     'a\\b.gguf', '*.gguf', 'a?.gguf', 'a[1].gguf', 'a\x00.gguf', 'model.bin', '', None):
            with self.subTest(name=name), self.assertRaises(ValueError): gguf_file([name], name)
        with self.assertRaises(ValueError): gguf_file(['a.gguf'], 'b.gguf')
        self.assertEqual(gguf_file(['folder/a.GGUF'], 'folder/a.GGUF'), 'folder/a.GGUF')

    def test_missing_file_never_starts_a_weight_download(self):
        info = SimpleNamespace(sha='sha', siblings=[SimpleNamespace(rfilename='other.gguf')])
        with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download') as fetch, patch('huggingface_hub.snapshot_download') as snapshot:
            api.return_value.model_info.return_value = info
            with self.assertRaises(ValueError): download('owner/model', filename='missing.gguf')
            fetch.assert_not_called(); snapshot.assert_not_called()

    def test_complete_shard_family_pins_every_download_without_other_variants(self):
        from dllm.download import gguf_files
        names = [f'Q4/model-0000{i}-of-00003.gguf' for i in range(1, 4)]
        files = names + ['Q8/model-00001-of-00003.gguf', 'Q4/other.gguf']
        self.assertEqual(gguf_files(files, names[0]), tuple(names))
        info = SimpleNamespace(sha='pinned-sha', siblings=[SimpleNamespace(rfilename=n) for n in files])
        with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download', side_effect=lambda repo, name, **kw: '/cache/'+name) as fetch, patch('huggingface_hub.snapshot_download') as snapshot:
            api.return_value.model_info.return_value = info
            result = download('owner/model', 'moving', names[0])
            self.assertEqual(result['path'], '/cache/'+names[0])
            self.assertEqual(result['filenames'], names)
            self.assertEqual([call.args[1] for call in fetch.call_args_list], names)
            self.assertTrue(all(call.kwargs == {'revision': 'pinned-sha'} for call in fetch.call_args_list))
            snapshot.assert_not_called()

    def test_missing_or_invalid_family_rejects_before_any_download(self):
        for name in ('model-00002-of-00002.gguf', 'model-00001-of-00065.gguf',
                     'model-00001-of-00000.gguf', 'model-00001-of-00001.gguf',
                     'model-00001-of-00002.gguf'):
            with self.subTest(name=name), patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download') as fetch:
                api.return_value.model_info.return_value = SimpleNamespace(sha='sha', siblings=[SimpleNamespace(rfilename=name)])
                with self.assertRaises(ValueError): download('owner/model', filename=name)
                fetch.assert_not_called()
