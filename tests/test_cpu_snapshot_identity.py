import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from damendaemon.cpu_snapshot_identity import FORMAT,fingerprint


class SnapshotIdentityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)
        for name,data in [('config.json',b'{}'),('tokenizer.json',b'{}'),('model.safetensors',b'fixture payload')]:
            (self.path/name).write_bytes(data)

    def test_independent_digest_and_relocation_control(self):
        records=[]
        for name in ['config.json','model.safetensors','tokenizer.json']:
            data=(self.path/name).read_bytes()
            records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        expected=hashlib.sha256(json.dumps({'format':FORMAT,'files':records},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result=fingerprint(self.path)
        self.assertEqual(result.model_id,expected);self.assertEqual(result.report()['files'],records)
        (self.path/'ignored.json').write_text('unrelated')
        self.assertEqual(fingerprint(self.path).model_id,expected);result.verify_unchanged()
        with tempfile.TemporaryDirectory() as other:
            for name in ['config.json','model.safetensors','tokenizer.json']:
                (Path(other)/name).write_bytes((self.path/name).read_bytes())
            self.assertEqual(fingerprint(other).model_id,expected)

    def test_mutation_inventory_and_same_size_replacement(self):
        result=fingerprint(self.path)
        file=self.path/'model.safetensors';before=file.stat()
        file.write_bytes(b'changed payload')
        os.utime(file,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'changed'):result.verify_unchanged()
        self.assertNotEqual(result.model_id,fingerprint(self.path).model_id)
        result=fingerprint(self.path)
        (self.path/'generation_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory'):result.verify_unchanged()
        self.assertNotEqual(result.model_id,fingerprint(self.path).model_id)

    def test_shard_index_and_path_containment(self):
        (self.path/'shards').mkdir();(self.path/'model.safetensors').rename(self.path/'shards/one.safetensors')
        index=self.path/'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map':{'weight':'shards/one.safetensors'}}))
        self.assertIn('shards/one.safetensors',[r[0] for r in fingerprint(self.path).files])
        for name in ['../escape.safetensors','/escape.safetensors','shards//one.safetensors']:
            index.write_text(json.dumps({'weight_map':{'weight':name}}))
            with self.assertRaises(ValueError):fingerprint(self.path)
        index.unlink()
        (self.path/'model.safetensors').symlink_to('shards/one.safetensors')
        with self.assertRaises(OSError):fingerprint(self.path)
        (self.path/'model.safetensors').unlink();os.mkfifo(self.path/'model.safetensors')
        with self.assertRaisesRegex(ValueError,'regular'):fingerprint(self.path)

    def test_bounds_deadline_and_missing_metadata(self):
        for kwargs in ({'max_bytes':1},{'max_bytes':True},{'timeout':float('nan')},{'timeout':float('inf')},{'timeout':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):fingerprint(self.path,**kwargs)
        with patch('damendaemon.cpu_snapshot_identity.time.monotonic',side_effect=[0,2]):
            with self.assertRaisesRegex(ValueError,'deadline'):fingerprint(self.path,timeout=1)
        (self.path/'tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError,'tokenizer'):fingerprint(self.path)
