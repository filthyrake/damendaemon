import json
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.split_gguf import execute, http_check, prepare, save, snapshot, verify_files
from tests.test_gguf import HAS_GGUF, fixture
from tests.test_gguf_split import shards
from tests.test_server import FakeRuntime


@unittest.skipUnless(HAS_GGUF, 'gguf dependency required')
class PreparationTests(unittest.TestCase):
    def test_real_writer_equivalence_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root/'single.gguf'; fixture(single)
            parts = shards(root)
            out = root/'evidence'
            manifest = prepare(single, parts[0], out)
            self.assertFalse(manifest['hardware_validated'])
            self.assertEqual(manifest['sources']['single']['semantics'], manifest['sources']['split']['semantics'])
            verify_files(manifest['sources']['split'])
            with parts[-1].open('r+b') as file:
                file.seek(-1, 2); value=file.read(1); file.seek(-1, 2); file.write(bytes([value[0]^1]))
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_files(manifest['sources']['split'])
            with self.assertRaisesRegex(ValueError, 'differ'):
                prepare(single, parts[0], root/'bad')

    def test_membership_and_option_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); single=root/'single.gguf'; fixture(single)
            parts=shards(root)
            for a,b in [(single,single),(parts[0],parts[0]),(single,parts[1])]:
                with self.subTest(a=a,b=b), self.assertRaises(ValueError):
                    prepare(a,b,root/'bad')
            for context,steps in [(31,4),(129,4),(128,0),(128,9)]:
                with self.assertRaises(ValueError):
                    prepare(single,parts[0],root/'bad',context,steps)
            self.assertFalse((root/'bad').exists())

    def test_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);single=root/'single.gguf';fixture(single)
            parts=shards(root,kind=1)
            with self.assertRaisesRegex(ValueError,'differ'):
                prepare(single,parts[0],root/'bad')


class OrchestrationTests(unittest.TestCase):
    def manifest(self, root):
        save(root/'manifest.json', dict(schema=1,context=128,steps=4,prompt_ids=[1],
                                       allocator_mib=1536,timeout_seconds=120))

    def test_sequential_phases_and_failure_evidence(self):
        for fail in (None,'split-trace','single-http'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.manifest(root);calls=[]
                def invoke(command, **kwargs):
                    self.assertEqual(command[2], 'scripts.experiments.split_gguf')
                    alias,phase=command[-2:];label=f'{alias}-{phase}';calls.append(label)
                    self.assertGreater(kwargs['timeout'],0)
                    self.assertLessEqual(kwargs['timeout'],120)
                    if label==fail:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
                    save(root/f'{label}.json',dict(result={'phase':phase}))
                if fail:
                    with self.assertRaises(subprocess.TimeoutExpired):execute(root,invoke)
                else:execute(root,invoke)
                report=json.loads((root/'result.json').read_text())
                self.assertEqual(report['passed'],fail is None)
                self.assertEqual(report['hardware_validated'],fail is None)
                self.assertFalse(report['performance_qualified'])
                expected=['single-trace','split-trace','single-http','split-http']
                self.assertEqual(calls,expected if fail is None else expected[:expected.index(fail)+1])
                with self.assertRaisesRegex(ValueError,'fresh'):execute(root,invoke)

    def test_final_worker_or_comparison_cannot_outlive_total_deadline(self):
        for stage in ('worker', 'comparison'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.manifest(root)
                now = [0.0]
                calls = []
                loads = json.loads
                comparisons = [0]
                def delayed_loads(raw):
                    value = loads(raw)
                    if value.get('result', {}).get('phase') == 'http':
                        comparisons[0] += 1
                        if stage == 'comparison' and comparisons[0] == 2:
                            now[0] = 121.0
                    return value
                def invoke(command, **kwargs):
                    alias, phase = command[-2:]
                    calls.append((alias, phase))
                    save(root/f'{alias}-{phase}.json', dict(result={'phase': phase}))
                    if stage == 'worker' and (alias, phase) == ('split', 'http'):
                        now[0] = 121.0
                with patch('scripts.experiments.split_gguf.time.monotonic', side_effect=lambda: now[0]), \
                        patch('scripts.experiments.split_gguf.json.loads', side_effect=delayed_loads):
                    with self.assertRaisesRegex(TimeoutError, 'total qualification deadline'):
                        execute(root, invoke)
                report = loads((root/'result.json').read_text())
                self.assertFalse(report['passed'])
                self.assertFalse(report['hardware_validated'])
                self.assertEqual(report['seconds'], 121.0)
                self.assertIn('total qualification deadline', report['error'])
                self.assertEqual(len(calls), 4)

    def test_mismatch_stops_before_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.manifest(root);calls=[]
            def invoke(command,**kwargs):
                alias,phase=command[-2:];calls.append(phase)
                save(root/f'{alias}-{phase}.json',dict(result=alias))
            with self.assertRaisesRegex(ValueError,'mismatch'):execute(root,invoke)
            self.assertEqual(calls,['trace','trace'])
            self.assertFalse(json.loads((root/'result.json').read_text())['passed'])

    def test_run_requires_guard(self):
        from scripts.experiments.split_gguf import main
        with patch.dict(os.environ,{},clear=True), patch('sys.argv',['runner','--run','unused']):
            with self.assertRaises(SystemExit):main()


class NumericalTraceTests(unittest.TestCase):
    def test_eos_nonfinite_and_cleanup(self):
        import torch
        from types import SimpleNamespace
        from scripts.experiments.split_gguf import trace
        class Executor:
            closed=False
            def prefill(self,ids):return torch.tensor([[0.,1.,2.]])
            def decode(self,token):raise AssertionError('must stop on EOS')
            def close(self):self.closed=True
        model=SimpleNamespace(eos={2})
        executor=Executor()
        with patch('dllm.runtime.make_executor',return_value=executor), \
             patch('dllm.prompts.decode_output',return_value=''):
            result=trace(model,[1],4)
        self.assertEqual(result['tokens'],[2])
        self.assertEqual(len(result['logits_sha256']),1)
        self.assertTrue(executor.closed)
        executor=Executor()
        with patch.object(executor,'prefill',return_value=torch.tensor([[float('nan')]])), \
             patch('dllm.runtime.make_executor',return_value=executor):
            with self.assertRaisesRegex(ValueError,'nonfinite'):trace(model,[1],4)
        self.assertTrue(executor.closed)


class HttpRunnerTests(unittest.TestCase):
    def test_real_loopback_lifecycle_and_mismatch_cleanup(self):
        class Runtime(FakeRuntime):
            loaded=False
            def manage(self,action,**options):
                if action=='load':
                    self.loaded=True
                    return dict(source_tier='disk')
                self.unloaded=True
                return dict(host_cached_bytes=0,models=[dict(residency='disk')])
            close_calls=0
            def close(self):self.close_calls+=1
            def generate(self,*args,**kwargs):
                if not self.loaded:raise AssertionError('HTTP generation before explicit cold load')
                for event in super().generate(*args,**kwargs):
                    if 'timing' in event:event['timing']['source_tier']='disk'
                    yield event
        for text in ('Hello world','wrong','truncated'):
            runtime=Runtime()
            with self.subTest(text=text):
                # FakeRuntime's timing already identifies disk; this checks the
                # real HTTP transport and runner assertions, not GPU correctness.
                if text == 'truncated':
                    from scripts.experiments import split_gguf
                    original = split_gguf.urlopen
                    def truncated(request, **kwargs):
                        response = original(request, **kwargs)
                        if json.loads(request.data).get('stream'):
                            with response:
                                return io.BytesIO(response.read().rstrip(b'\n'))
                        return response
                    with patch.object(split_gguf, 'urlopen', side_effect=truncated):
                        with self.assertRaisesRegex(ValueError, 'unterminated SSE event'):
                            http_check(runtime, [1], 4, dict(text='Hello world', completion_tokens=2))
                elif text=='wrong':
                    with self.assertRaisesRegex(ValueError,'trace'):
                        http_check(runtime,[1],4,dict(text=text,completion_tokens=2))
                else:
                    result=http_check(runtime,[1],4,dict(text=text,completion_tokens=2))
                    self.assertTrue(result['disk_reload'])
                    self.assertTrue(runtime.unloaded)
                self.assertEqual(runtime.close_calls,1)


if __name__=='__main__':unittest.main()
