"""Real upstream-writer shards, compared with independent single-file import."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tests.test_gguf import HAS_GGUF, fixture

if HAS_GGUF:
    from dllm import gguf_model as importer
    from dllm.gguf_reader import BoundedReader


def shards(folder, **kwargs):
    fixture(Path(folder)/'split.gguf', split_max_tensors=2, **kwargs)
    return sorted(Path(folder).glob('split-*.gguf'))


def replace_integer(path, key, value):
    with BoundedReader(path, allow_empty=True) as reader:
        part = reader.fields[key].parts[-1]
        offset = part.ctypes.data - reader.data.ctypes.data
        raw = value.to_bytes(part.dtype.itemsize, 'little', signed=part.dtype.kind == 'i')
    with path.open('r+b') as file:
        file.seek(offset); file.write(raw)


@unittest.skipUnless(HAS_GGUF, 'optional gguf dependency is not installed')
class SplitGgufTests(unittest.TestCase):
    def test_registry_and_manifest_preserve_opaque_shard_namespace(self):
        from dllm.runtime import Runtime
        from dllm.model_manifest import snapshot, save, load
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root/'source'; source.mkdir()
            cache = root/'snapshot'; cache.mkdir()
            blobs = root/'blobs'; blobs.mkdir()
            parts = shards(source)
            for index, part in enumerate(parts):
                blob = blobs/f'opaque{index}'; blob.write_bytes(part.read_bytes())
                (cache/part.name).symlink_to(blob)
            selected = cache/parts[0].name
            with patch('dllm.runtime.require_gpu'), patch.object(Runtime, '_status', return_value={}):
                runtime = Runtime({}, context=64)
                try:
                    runtime.manage('register', 'split', str(selected), quantization='none')
                    runtime.manage('register', 'split', str(selected), quantization='none')
                    self.assertEqual(runtime.models['split'], selected)
                    exported = snapshot(runtime.models, runtime.quantizations, runtime.chat_templates)
                    manifest = root/'models.json'; save(manifest, exported)
                    models, _, _ = load(manifest, lambda *_: self.fail('unexpected download'))
                    self.assertEqual(models['split'], selected)
                    self.assertEqual(importer.plan(models['split']).config, importer.plan(parts[0]).config)
                    self.assertGreater(importer.GgufWeights(models['split'], 64, 'none').bytes, 0)
                finally:
                    runtime.close()

    def test_exact_single_file_equivalence(self):
        for kind in (0, 1, 2, 8, 12, 13, 14):
            for tied in (False, True):
                with self.subTest(kind=kind, tied=tied), tempfile.TemporaryDirectory() as folder:
                    options = dict(kind=kind, tied=tied, packed_embedding=True,
                                   hidden=256 if kind >= 12 else 32, mixed=kind in (8, 12))
                    single = Path(folder)/'single.gguf'; fixture(single, **options)
                    parts = shards(folder, **options)
                    a, b = (importer.GgufWeights(p, 64, 'none') for p in (single, parts[0]))
                    self.assertEqual(a.config, b.config)
                    self.assertEqual(a.tokenizer.to_str(), b.tokenizer.to_str())
                    self.assertEqual(a.eos, b.eos)
                    self.assertEqual(a.packed_formats, b.packed_formats)
                    self.assertEqual(a.bytes, b.bytes)
                    self.assertEqual(importer.weight_bytes(single, 'none'), importer.weight_bytes(parts[0], 'none'))
                    for collection in ('weights', 'scales'):
                        left, right = getattr(a, collection), getattr(b, collection)
                        self.assertEqual(left.keys(), right.keys())
                        for name in left: self.assertTrue(torch.equal(left[name], right[name]), name)
                    if tied: self.assertIs(b.weights['embed'], b.weights['head'])
                    source = importer.plan(parts[0])
                    self.assertGreater(len({source.tensors[f'blk.0.attn_{x}.weight'].file_index for x in ('q','k','v')}), 1)

    def test_metadata_only_first_shard_and_closed_mappings(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder, small_first_shard=True)
            mappings = []
            class ObservedReader(BoundedReader):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs); mappings.append(self.data._mmap)
            with patch.object(importer, 'BoundedReader', ObservedReader):
                source = importer.plan(parts[0])
            self.assertTrue(mappings)
            self.assertTrue(all(m.closed for m in mappings))
            self.assertTrue(all(t.file_index > 0 for t in source.tensors.values()))
            self.assertIs(source, importer.plan(parts[0]))
            self.assertGreater(importer.GgufWeights(parts[0], 64, 'none').bytes, 0)

    def test_later_shard_cache_invalidation_and_missing_members(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            original = importer.plan(parts[0])
            info = parts[-1].stat()
            os.utime(parts[-1], ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
            self.assertIsNot(original, importer.plan(parts[0]))
            parts[-1].unlink()
            with self.assertRaises(FileNotFoundError): importer.plan(parts[0])

    def test_malformed_membership_and_nonfirst_selection(self):
        for key, value, index in (('split.no', 0, 1), ('split.count', 65, 0),
                                  ('split.tensors.count', 1, 1), ('split.tensors.count', 1, 0)):
            with self.subTest(key=key, index=index), tempfile.TemporaryDirectory() as folder:
                parts = shards(folder)
                replace_integer(parts[index], key, value)
                with self.assertRaisesRegex(ValueError, 'split'): importer.plan(parts[0])
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            with self.assertRaisesRegex(ValueError, 'first GGUF shard'): importer.plan(parts[1])
            renamed = parts[0].with_name('renamed.gguf'); parts[0].rename(renamed)
            with self.assertRaisesRegex(ValueError, 'first GGUF shard'): importer.plan(renamed)

    def test_duplicate_tensor_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            # Q and K names have the same byte length, on different shards.
            for path in parts:
                raw = path.read_bytes()
                if b'blk.0.attn_k.weight' in raw:
                    path.write_bytes(raw.replace(b'blk.0.attn_k.weight', b'blk.0.attn_q.weight'))
                    break
            with patch.object(importer, 'read_tensor', side_effect=AssertionError('payload read')):
                with self.assertRaisesRegex(ValueError, 'duplicate tensor'): importer.GgufWeights(parts[0], 64, 'none')

    def test_changed_payload_discards_model_and_closes_all_files(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            importer.plan(parts[0])
            opened = []
            resolved_parts = {p.resolve() for p in parts}
            original_open, original_read = Path.open, importer.read_tensor
            changed = False
            def observe(path, *args, **kwargs):
                file = original_open(path, *args, **kwargs)
                if path in resolved_parts: opened.append(file)
                return file
            def read(file, tensor):
                nonlocal changed
                if not changed:
                    changed = True
                    info = parts[-1].stat()
                    os.utime(parts[-1], ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
                return original_read(file, tensor)
            with patch.object(Path, 'open', observe), patch.object(importer, 'read_tensor', read):
                with self.assertRaisesRegex(ValueError, 'changed'): importer.GgufWeights(parts[0], 64, 'none')
            self.assertEqual(len(opened), len(parts))
            self.assertTrue(all(file.closed for file in opened))

    def test_aggregate_header_bound_and_conflicting_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            with patch.object(importer, 'MAX_HEADER', 1):
                with self.assertRaisesRegex(ValueError, 'aggregate'): importer.plan(parts[0])
            class ConflictingReader(BoundedReader):
                def __init__(self, path, **kwargs):
                    super().__init__(path, **kwargs)
                    if Path(path).name == parts[1].name:
                        from unittest.mock import Mock
                        self.fields['general.architecture'] = Mock(contents=lambda: 'llama')
            with patch.object(importer, 'BoundedReader', ConflictingReader):
                with self.assertRaisesRegex(ValueError, 'metadata disagrees'): importer.plan(parts[0])

    def test_payload_failure_closes_previously_opened_files(self):
        with tempfile.TemporaryDirectory() as folder:
            parts = shards(folder)
            importer.plan(parts[0])
            opened = []; original_open = Path.open
            paths = {p.resolve() for p in parts}
            def fail(path, *args, **kwargs):
                if path == parts[-1].resolve(): raise OSError('injected open failure')
                file = original_open(path, *args, **kwargs)
                if path in paths: opened.append(file)
                return file
            with patch.object(Path, 'open', fail):
                with self.assertRaisesRegex(OSError, 'injected'): importer.GgufWeights(parts[0], 64, 'none')
            self.assertEqual(len(opened), len(parts) - 1)
            self.assertTrue(all(file.closed for file in opened))

    def test_opaque_cache_links_preserve_sibling_namespace_and_invalidation(self):
        from dllm.cli import resolve_model
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            materialized = root/'source'; materialized.mkdir()
            parts = shards(materialized)
            cache = root/'snapshot'; cache.mkdir()
            blobs = root/'blobs'; blobs.mkdir()
            links = []
            for index, path in enumerate(parts):
                blob = blobs/f'opaque{index}'; blob.write_bytes(path.read_bytes())
                link = cache/path.name; link.symlink_to(blob); links.append(link)
            selected = resolve_model(str(links[0]))
            self.assertEqual(selected, links[0])
            a = importer.GgufWeights(parts[0], 64, 'none')
            b = importer.GgufWeights(selected, 64, 'none')
            for name in a.weights: self.assertTrue(torch.equal(a.weights[name], b.weights[name]))
            source = importer.plan(selected)
            replacement = blobs/'new-target'; replacement.write_bytes(links[-1].read_bytes())
            links[-1].unlink(); links[-1].symlink_to(replacement)
            self.assertIsNot(source, importer.plan(selected))
            # Swapping the target while a payload handle is open must fail the
            # final namespace check, even if the old open file stays unchanged.
            read = importer.read_tensor; changed = False
            def retarget(file, tensor):
                nonlocal changed
                if not changed:
                    changed = True
                    links[-1].unlink(); links[-1].symlink_to(blobs/f'opaque{len(parts)-1}')
                return read(file, tensor)
            with patch.object(importer, 'read_tensor', retarget):
                with self.assertRaisesRegex(ValueError, 'changed'): importer.GgufWeights(selected, 64, 'none')
