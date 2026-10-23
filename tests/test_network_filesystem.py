"""Filesystem-class identity tests: network mounts tolerate ctime jitter only.

Local files keep the strict size/mtime/ctime receipt everywhere. A local
temporary file whose ctime is touched with os.utime (same atime/mtime)
simulates the read-side ctime jitter network filesystems exhibit, so the two
modes are distinguishable without depending on a mounted share.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from damendaemon import network_filesystem
from damendaemon.network_filesystem import network_device
from damendaemon import gguf_model
from damendaemon import awq_model


def _touch_ctime_only(path):
    info = os.stat(path)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


class _TempCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        network_filesystem._network_devices.cache_clear()
        self.addCleanup(network_filesystem._network_devices.cache_clear)

    def _file(self, name='model.safetensors', data=b'fixture payload'):
        target = self.path / name
        target.write_bytes(data)
        return target


class MountTableTests(_TempCase):
    def test_decode_mount_name_unescapes_proc_mounts(self):
        self.assertEqual(network_filesystem._decode_mount_name('a\\040b\\011c'), 'a b\tc')
        self.assertEqual(network_filesystem._decode_mount_name('d\\012e\\015f\\134g'), 'd\ne\rf\\g')

    def test_macos_mount_table_parses_smb_and_nfs(self):
        text = ('//user@nas.local/llm-models on /Volumes/llm-models (smbfs, nodev, nosuid, '
                'mounted by damenknight)\nnas.local:/export on /mnt/nas (nfs, read-only)\n'
                '/dev/disk3s1 on / (apfs, local)\n')
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=text)
        with patch('subprocess.run', return_value=completed):
            table = network_filesystem._macos_mount_table()
        self.assertEqual(table, [('/Volumes/llm-models', 'smbfs'), ('/mnt/nas', 'nfs'), ('/', 'apfs')])

    def test_macos_mount_table_survives_mount_failure(self):
        with patch('subprocess.run', side_effect=OSError('no mount')):
            self.assertEqual(network_filesystem._macos_mount_table(), [])

    def test_mount_table_prefers_proc_mounts(self):
        with patch.object(os.path, 'exists', return_value=True), \
                patch.object(network_filesystem, '_linux_mount_table', return_value=[('/x', 'nfs')]):
            self.assertEqual(network_filesystem._mount_table(), [('/x', 'nfs')])

    def test_devices_map_stat_of_network_mountpoints_only(self):
        local = self._file('local.txt', b'x')
        with patch.object(network_filesystem, '_mount_table',
                          return_value=[(str(self.path), 'nfs'), ('/unavailable', 'smbfs')]):
            devices = network_filesystem._network_devices()
        self.assertIn(os.stat(local).st_dev, devices)
        self.assertTrue(network_device(os.stat(local).st_dev))
        self.assertFalse(network_device(0x7fffffff7fffffff))

    def test_is_network_filesystem_local_false_and_unreadable_false(self):
        target = self._file()
        self.assertFalse(network_filesystem.is_network_filesystem(target))
        self.assertFalse(network_filesystem.is_network_filesystem(self.path / 'missing'))


@unittest.skipUnless(os.path.isdir('/Volumes/llm-models'), 'real SMB share not mounted')
class RealShareTests(_TempCase):
    def test_real_share_classified_network_and_local_stays_strict(self):
        self.assertTrue(network_filesystem.is_network_filesystem('/Volumes/llm-models'))
        local = self._file()
        self.assertFalse(network_filesystem.is_network_filesystem(local))


class IdentityModeTests(_TempCase):
    def test_gguf_strict_local_ctime_is_load_bearing(self):
        target = self._file('model.gguf', b'gguf')
        before = gguf_model.identity(target)
        _touch_ctime_only(target)
        after = gguf_model.identity(target)
        self.assertEqual(len(before), 5)
        self.assertNotEqual(before, after)

    def test_gguf_network_mode_tolerates_ctime_jitter(self):
        target = self._file('model.gguf', b'gguf')
        with patch('damendaemon.gguf_model.network_device', return_value=True):
            before = gguf_model.identity(target)
            _touch_ctime_only(target)
            after = gguf_model.identity(target)
        self.assertEqual(len(before), 4)
        self.assertEqual(before, after)

    def test_safetensor_stamp_strict_local(self):
        from damendaemon.safetensor_header import file_stamp
        target = self._file(data=b'x' * 64)
        with open(target, 'rb') as reader:
            before = file_stamp(reader)
            _touch_ctime_only(target)
            after = file_stamp(reader)
        self.assertEqual(len(before), 3)
        self.assertNotEqual(before, after)

    def test_safetensor_stamp_network_mode_and_header_read(self):
        from damendaemon.safetensor_header import file_stamp, read_header_json
        header = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        payload = b'\x00\x00\x80?'
        target = self._file(data=struct.pack('<Q', len(header)) + header + payload)
        with patch('damendaemon.safetensor_header.network_device', return_value=True):
            with open(target, 'rb') as reader:
                before = file_stamp(reader)
                read_header_json(reader)
                _touch_ctime_only(target)
                self.assertEqual(file_stamp(reader), before)
            with open(target, 'rb') as reader:
                header_json, offset, stamp = read_header_json(reader)
                self.assertEqual(len(stamp), 2)

    def test_cpu_snapshot_strict_local_detects_jitter(self):
        from damendaemon.cpu_snapshot_identity import fingerprint
        for name in ('config.json', 'tokenizer.json'):
            (self.path / name).write_bytes(b'{}')
        self._file()
        identity = fingerprint(self.path)
        _touch_ctime_only(self.path / 'config.json')
        with self.assertRaises(ValueError):
            identity.verify_unchanged()

    def test_cpu_snapshot_network_mode_tolerates_jitter_detects_size(self):
        from damendaemon.cpu_snapshot_identity import fingerprint
        for name in ('config.json', 'tokenizer.json'):
            (self.path / name).write_bytes(b'{}')
        shard = self._file()
        with patch('damendaemon.cpu_snapshot_identity.network_device', return_value=True):
            identity = fingerprint(self.path)
            _touch_ctime_only(self.path / 'config.json')
            identity.verify_unchanged()
            (self.path / 'config.json').write_bytes(b'{"a":1}')
            with self.assertRaises(ValueError):
                identity.verify_unchanged()
        self.assertNotEqual(os.stat(shard).st_size, 0)


class InventoryModeTests(_TempCase):
    def test_reference_pipeline_inventory_strict_and_network(self):
        from damendaemon.reference_pipeline_build import snapshot_inventory
        (self.path / 'config.json').write_bytes(b'{}')
        shard = self._file()
        strict = snapshot_inventory(self.path)
        self.assertTrue(all(len(record) == 6 for record in strict))
        with patch('damendaemon.reference_pipeline_build.network_device', return_value=True):
            network = snapshot_inventory(self.path)
        self.assertTrue(all(len(record) == 5 for record in network))
        shard.write_bytes(b'fixture payload changed')
        self.assertNotEqual(snapshot_inventory(self.path), strict)
        with patch('damendaemon.reference_pipeline_build.network_device', return_value=True):
            self.assertNotEqual(snapshot_inventory(self.path), network)

    def test_awq_identity_strict_and_network(self):
        target = self._file('model.safetensors', b'x' * 32)
        strict = awq_model.identity(target)
        assert strict is not None
        self.assertEqual(len(strict), 5)
        _touch_ctime_only(target)
        self.assertNotEqual(awq_model.identity(target), strict)
        with patch('damendaemon.awq_model.network_device', return_value=True):
            network = awq_model.identity(target)
            assert network is not None
            self.assertEqual(len(network), 4)
            _touch_ctime_only(target)
            self.assertEqual(awq_model.identity(target), network)

    def test_weight_cache_snapshot_strict_and_network(self):
        from damendaemon.weight_cache import PreparedCache
        (self.path / 'config.json').write_bytes(b'{}')
        self._file()
        cache_root = self.path / 'cache'; cache_root.mkdir()
        model = self.path
        strict = PreparedCache(cache_root, model, {'model_type': 'qwen2'}, 'q4_0',
                               [model / 'model.safetensors'])
        self.assertTrue(all(len(record) == 6 for record in strict.identity))
        with patch('damendaemon.weight_cache.network_device', return_value=True):
            network = PreparedCache(cache_root, model, {'model_type': 'qwen2'}, 'q4_0',
                                    [model / 'model.safetensors'])
        self.assertTrue(all(len(record) == 5 for record in network.identity))


if __name__ == '__main__':
    unittest.main()
