"""#674 cell 5 CPU contracts: the --auto-tune pre-serve calibration.

The orchestration (calibrate) is covered with injected enumeration,
measurement and corpus seams — no GPU, checkpoint or tokenizer — pinning:
the winner selection, artifact emission (rows retained on failure,
ranking, serve-consumable model-config), the applied-parameters record
the CLI applies, refusals for unhonorable manifests, and the honest
all-candidates-failed refusal to start.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from damendaemon.autotune import Candidate
from damendaemon.autotune_serve import (calibrate, row_from_measurement)


def tune_manifest(**overrides):
    ids = list(range(4096))
    manifest = {
        'schema_version': 1,
        'model': {'alias': 'model', 'path': '/tmp/fake-model', 'quantization': 'q8_0',
                  'mtp': False},
        'hardware': {'accelerator': 'cuda'},
        'context': 2048,
        'corpus': {'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   'prompt_tokens': len(ids), 'description': 'pinned'},
        'requests': {'count': 4, 'max_tokens': 64, 'concurrency': 2},
        'objective': {'metric': 'aggregate_output_tokens_per_second',
                      'direction': 'maximize',
                      'gates': {'zero_errors': True, 'exact_completion_tokens': 64}},
        'search': {'rounds': 1, 'timeout_seconds': 600, 'values': {}},
    }
    manifest.update(overrides)
    return manifest


class Fake:
    """The injected seams: two admitted candidates (batch 2 vs 4) plus one
    refused, and a measurement keyed on the candidate's batch."""
    CANDIDATES = [Candidate(quantization='q8_0', precision='float16',
                           kernels='triton', graphs=False, max_batch=2),
                  Candidate(quantization='q8_0', precision='float16',
                           kernels='triton', graphs=False, max_batch=4)]

    @classmethod
    def enumerate_fn(cls, config, *, accelerator, context, has_mtp, grid):
        refused = Candidate(quantization='q4_0', precision='compensated',
                            kernels='triton', graphs=False, max_batch=1)
        return list(cls.CANDIDATES), [(refused, 'experimental hybrid Q4 requires '
                                            'explicit opt-in')]

    @staticmethod
    def corpus_fn(path, prompt_tokens, digest):
        ids = list(range(prompt_tokens))
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == digest
        return ids

    @classmethod
    def measure_fn(cls, metrics=None, failures=()):
        metrics = metrics or {2: 10.0, 4: 25.0}
        def measure(candidate, manifest, model_path, corpus_ids):
            if candidate.max_batch in failures:
                return row_from_measurement(
                    candidate, None, None, None, 0,
                    manifest['objective']['metric'],
                    unmeasurable=f'RuntimeError: simulated {candidate.max_batch}')
            # tokens/round_seconds == the declared aggregate metric.
            tokens = [64, 64]
            round_seconds = sum(tokens) / metrics[candidate.max_batch]
            return row_from_measurement(
                candidate, [1.0, 1.0], [3.0, 3.0], tokens, round_seconds,
                manifest['objective']['metric'])
        return measure


class CalibrateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'model').mkdir()
        self.manifest_path = self.root / 'tune.json'
        self.output = self.root / 'tuned'
        self.manifest = tune_manifest(
            model={'alias': 'model', 'path': str(self.root / 'model'),
                   'quantization': 'q8_0', 'mtp': False})

    def write_manifest(self, **overrides):
        manifest = tune_manifest(**overrides)
        manifest['model']['path'] = str(self.root / 'model')
        self.manifest_path.write_text(json.dumps(manifest))
        return manifest

    def calibrate(self, **kwargs):
        params = dict(measure=Fake.measure_fn(), enumerate_fn=Fake.enumerate_fn,
                     corpus_fn=Fake.corpus_fn)
        params.update(kwargs)
        return calibrate(self.manifest_path, self.output,
                         {'model': self.root / 'model'}, 2048, 'cuda', **params)

    def test_winner_selection_and_artifacts(self):
        self.write_manifest()
        record, applied = self.calibrate()
        # Batch 4 wins on the declared objective (25 > 10 aggregate).
        self.assertEqual(record['metric_value'], 25.0)
        self.assertEqual(applied['batch_size'], 4)
        self.assertTrue(applied['use_graph'] is False)
        self.assertEqual(applied['environment']['DAMENDAEMON_HYBRID_PRECISION'],
                         'float16')
        self.assertEqual(applied['queue_size'], 8)
        # Artifacts: per-candidate rows, the refusal record, the ranking,
        # and the serve-consumable model-config.
        names = sorted(p.name for p in self.output.iterdir())
        self.assertEqual(names, ['ranking.json', 'ranking.model-config.json',
                                 'refusals.json', 'row-candidate-0.json',
                                 'row-candidate-1.json'])
        ranking = json.loads((self.output / 'ranking.json').read_text())
        self.assertEqual([row['candidate']['max_batch'] for row in ranking['rows_ranked']],
                         [4, 2])
        self.assertEqual(ranking['winner']['metric_value'], 25.0)
        model_config = json.loads((self.output / 'ranking.model-config.json').read_text())
        self.assertEqual(model_config['version'], 1)
        self.assertEqual(model_config['models']['model']['path'],
                         str(self.root / 'model'))
        refusals = json.loads((self.output / 'refusals.json').read_text())
        self.assertEqual(len(refusals), 1)
        self.assertIn('explicit opt-in', refusals[0]['refusal'])

    def test_failed_candidate_is_retained_and_ranks_out(self):
        self.write_manifest()
        record, applied = self.calibrate(
            measure=Fake.measure_fn(failures={4}))
        # Batch 4 failed its measurement: disclosed, ranked out; batch 1 wins.
        self.assertEqual(record['metric_value'], 10.0)
        self.assertEqual(applied['batch_size'], 2)
        row = json.loads((self.output / 'row-candidate-1.json').read_text())
        self.assertIn('simulated 4', row['unmeasurable_reason'])
        ranking = json.loads((self.output / 'ranking.json').read_text())
        self.assertEqual(len(ranking['rows_eliminated']), 1)

    def test_all_candidates_failed_refuses_to_start(self):
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, 'no candidate passed'):
            self.calibrate(measure=Fake.measure_fn(failures={2, 4}))
        ranking = json.loads((self.output / 'ranking.json').read_text())
        self.assertIsNone(ranking['winner'])
        # Rows are still retained — the evidence outlives the refusal.
        self.assertTrue((self.output / 'row-candidate-0.json').exists())

    def test_gate_failure_ranks_out_not_refuses(self):
        # A candidate that measures but violates the exact-token gate is
        # eliminated with reasons, not silently dropped.
        self.write_manifest()
        def measure(candidate, manifest, model_path, corpus_ids):
            return row_from_measurement(candidate, [1.0], [3.0], [63], 6.0,
                                       manifest['objective']['metric'])
        with self.assertRaisesRegex(RuntimeError, 'no candidate passed'):
            self.calibrate(measure=measure)

    def test_manifest_must_match_the_serve(self):
        self.write_manifest()
        # Alias not served.
        with self.assertRaisesRegex(ValueError, 'not served'):
            self.calibrate_manifest = calibrate(
                self.manifest_path, self.output, {'other': self.root / 'model'},
                2048, 'cuda', measure=Fake.measure_fn(),
                enumerate_fn=Fake.enumerate_fn, corpus_fn=Fake.corpus_fn)
        # Path mismatch.
        with self.assertRaisesRegex(ValueError, 'does not match'):
            calibrate(self.manifest_path, self.output,
                      {'model': self.root / 'elsewhere'}, 2048, 'cuda',
                      measure=Fake.measure_fn(), enumerate_fn=Fake.enumerate_fn,
                      corpus_fn=Fake.corpus_fn)
        # Context mismatch.
        with self.assertRaisesRegex(ValueError, 'context'):
            self.write_manifest()
            calibrate(self.manifest_path, self.output,
                      {'model': self.root / 'model'}, 4096, 'cuda',
                      measure=Fake.measure_fn(), enumerate_fn=Fake.enumerate_fn,
                      corpus_fn=Fake.corpus_fn)
        # Device mismatch.
        with self.assertRaisesRegex(ValueError, 'accelerator'):
            calibrate(self.manifest_path, self.output,
                      {'model': self.root / 'model'}, 2048, 'rocm',
                      measure=Fake.measure_fn(), enumerate_fn=Fake.enumerate_fn,
                      corpus_fn=Fake.corpus_fn)

    def test_output_directory_must_be_fresh(self):
        self.write_manifest()
        self.output.mkdir()
        (self.output / 'stale.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'empty or absent'):
            self.calibrate()

    def test_concurrency_must_fit_every_candidate(self):
        self.write_manifest(requests={'count': 4, 'max_tokens': 64,
                                       'concurrency': 8})
        with self.assertRaisesRegex(ValueError, 'concurrency'):
            self.calibrate()

    def test_empty_search_refuses(self):
        class EmptyEnumerate:
            def __call__(self, config, *, accelerator, context, has_mtp, grid):
                return [], []
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, 'admits no candidates'):
            self.calibrate(enumerate_fn=EmptyEnumerate())


class FlagWiringTests(unittest.TestCase):
    """Real CLI subprocess checks: the flag surface refuses misuse before
    any calibration side effect."""

    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, '-m', 'damendaemon', 'serve', *extra],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, 'DAMENDAEMON_API_KEY': 'x' * 16})

    def test_auto_tune_requires_output_dir(self):
        result = self.run_cli('--model', 'm=/nope', '--auto-tune', '/nope.json')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--auto-tune requires --auto-tune-output', result.stderr)

    def test_auto_tune_requires_single_model_config(self):
        result = self.run_cli('--model', 'a=/nope', '--model', 'b=/nope',
                              '--auto-tune', '/nope.json',
                              '--auto-tune-output', '/tmp/never')
        self.assertEqual(result.returncode, 2)
        self.assertIn('exactly one served model', result.stderr)

    def test_missing_manifest_file_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            served = Path(tmp) / 'served.json'
            served.write_text(json.dumps(
                {'version': 1, 'models': {'model': dict(path=f'{tmp}/model',
                                                        quantization='q8_0')}}))
            result = self.run_cli('--model-config', str(served),
                                  '--auto-tune', f'{tmp}/absent.json',
                                  '--auto-tune-output', f'{tmp}/out')
            self.assertEqual(result.returncode, 2)
            self.assertIn('auto-tune calibration refused', result.stderr)


if __name__ == '__main__':
    unittest.main()
