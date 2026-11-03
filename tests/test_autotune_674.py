"""#674 auto-tune CPU contracts: the admission-aware enumerator, the workload
manifest, the objective predicate, and the winner's serve-consumable output.

The enumeration tests run against the REAL qwen3_5_text policy family with
the accelerator simulated through the same DAMENDAEMON_DEVICE surface the
policies read — no GPU is touched, and every admitted/refused decision is
produced by the policy functions themselves, never re-implemented here.
"""
import json
import os
import unittest
from unittest import mock

from damendaemon.autotune import (
    Candidate, Measurement, admit, corpus_digest, declared_grid,
    enumerate_candidates, evaluate_gates, parse_manifest, rank,
    serve_manifest, winner_record,
)

QWEN35_TEXT_CONFIG = {
    'model_type': 'qwen3_5_text',
    'num_hidden_layers': 36,
    'layer_types': ['linear_attention' if i % 4 != 3 else 'full_attention'
                    for i in range(36)],
}


def cuda_manifest():
    return {
        'schema_version': 1,
        'model': {'alias': 'qwen36', 'path': '/models/Qwen3.6-35B-A3B',
                  'quantization': 'q8_0', 'mtp': False},
        'hardware': {'accelerator': 'cuda'},
        'context': 32768,
        'corpus': {'prompt_sha256': 'a' * 64, 'prompt_tokens': 32256,
                   'description': 'pinned C-series paragraph'},
        'requests': {'count': 6, 'max_tokens': 512, 'concurrency': 6},
        'objective': {'metric': 'aggregate_output_tokens_per_second',
                      'direction': 'maximize',
                      'gates': {'max_first_output_seconds': 600.0,
                                'max_request_seconds': 1200,
                                'exact_completion_tokens': 512,
                                'zero_errors': True}},
        'search': {'rounds': 2, 'timeout_seconds': 1800, 'values': {}},
    }


def admit_q35(candidate, **kwargs):
    kwargs.setdefault('accelerator', 'cuda')
    kwargs.setdefault('context', 32768)
    kwargs.setdefault('has_mtp', False)
    return admit(candidate, QWEN35_TEXT_CONFIG, **kwargs)


class AdmissionEnumerationTests(unittest.TestCase):
    def test_production_surface_is_enumerated_from_the_real_policy(self):
        # The C-series shape: batch sweep is a first-class grid dimension.
        admitted, refusals = enumerate_candidates(
            QWEN35_TEXT_CONFIG, accelerator='cuda', context=32768, has_mtp=False,
            extra_values={'max_batch': (1, 4, 6, 8), 'kernels': ('triton', 'torch'),
                          'prefill': ('scan',), 'mtp': ('dropped',),
                          'mtp_spec_graph': (False,)})
        self.assertTrue(admitted)
        for candidate in admitted:
            self.assertNotEqual(candidate.quantization, 'q4_0')
            self.assertEqual(candidate.mtp, 'dropped')
        production = Candidate(quantization='q8_0', precision='float16',
                               kernels='triton', graphs=True, max_batch=6)
        self.assertIn(production, admitted)
        # Every refusal is the policy's own text, retained for review.
        self.assertTrue(refusals)
        for _, refusal in refusals:
            self.assertTrue(refusal)

    def test_q4_refusals_carry_the_policy_s_own_text(self):
        candidate = Candidate(quantization='q4_0', precision='compensated',
                              kernels='triton', graphs=False, max_batch=1)
        ok, refusal = admit_q35(candidate)
        self.assertFalse(ok)
        # The exact hybrid_q4_policy refusal, not a paraphrase.
        self.assertIn('experimental hybrid Q4 requires explicit opt-in', refusal)

    def test_compensated_q8_on_cuda_refuses_with_the_policy_text(self):
        candidate = Candidate(quantization='q8_0', precision='compensated',
                              kernels='triton', graphs=False, max_batch=1)
        ok, refusal = admit_q35(candidate)
        self.assertFalse(ok)
        self.assertIn('admitted packed-text device', refusal)

    def test_float32_packed_surface_and_the_four_slot_cap(self):
        # On CUDA the float32 packed surface is structurally empty without
        # the Q4 opt-in (q8 float32 is ROCm-scoped); on ROCm it is admitted
        # with the policy's own four-slot cap.
        ok, refusal = admit_q35(Candidate(quantization='q8_0', precision='float32',
                                          kernels='triton', graphs=False, max_batch=4))
        self.assertFalse(ok)
        self.assertIn('admitted packed-text device', refusal)
        rocm = dict(accelerator='rocm', context=512, has_mtp=False)
        self.assertTrue(admit(Candidate(quantization='q8_0', precision='float32',
                                       kernels='triton', graphs=False, max_batch=4),
                              QWEN35_TEXT_CONFIG, **rocm)[0])
        ok, refusal = admit(Candidate(quantization='q8_0', precision='float32',
                                      kernels='triton', graphs=False, max_batch=8),
                            QWEN35_TEXT_CONFIG, **rocm)
        self.assertFalse(ok)
        self.assertIn('at most four slots', refusal)
        ok, refusal = admit(Candidate(quantization='q8_0', precision='float32',
                                      kernels='triton', graphs=True, max_batch=4),
                            QWEN35_TEXT_CONFIG, **rocm)
        self.assertFalse(ok)
        self.assertIn('--no-graph', refusal)

    def test_mtp_gates_use_the_constructor_s_verbatim_texts(self):
        head = dict(quantization='q8_0', precision='float16', kernels='triton',
                    max_batch=4)
        ok, refusal = admit_q35(Candidate(mtp='on', graphs=True, **head), has_mtp=True)
        self.assertFalse(ok)
        self.assertEqual(refusal, 'the MTP draft head requires eager decode '
                                  '(no CUDA graphs) without the spec-graph opt-in')
        ok, refusal = admit_q35(Candidate(mtp='on', kv_quant='q8_0',
                                          graphs=False, **head), has_mtp=True)
        self.assertFalse(ok)
        self.assertEqual(refusal, 'the MTP draft head is not qualified with quantized KV')
        # The head must exist in the checkpoint at all.
        ok, refusal = admit_q35(Candidate(mtp='on', graphs=False, **head), has_mtp=False)
        self.assertFalse(ok)
        self.assertIn('requires a checkpoint', refusal)
        # With the head present, eager MTP is admitted.
        ok, _ = admit_q35(Candidate(mtp='on', graphs=False, **head), has_mtp=True)
        self.assertTrue(ok)
        # The #691 opt-in is inert on CUDA production surfaces (compensated
        # q8 refuses at the precision gate first; float16 never captures).
        ok, refusal = admit_q35(Candidate(mtp='on', mtp_spec_graph=True, graphs=True,
                                          precision='compensated',
                                          quantization='q8_0', kv_quant='none',
                                          max_batch=1), has_mtp=True)
        self.assertFalse(ok)
        self.assertIn('admitted packed-text device', refusal)
        ok, refusal = admit_q35(Candidate(mtp='on', mtp_spec_graph=True, graphs=True,
                                          precision='float16', quantization='q8_0',
                                          max_batch=1), has_mtp=True)
        self.assertFalse(ok)
        self.assertIn('spec-graph opt-in requires', refusal)

    def test_kernel_and_prefill_executor_gates(self):
        ok, refusal = admit_q35(Candidate(quantization='q8_0', precision='float16',
                                          kernels='torch', graphs=True, max_batch=1),
                                 quantized_kernels_import=False)
        self.assertFalse(ok)
        self.assertEqual(refusal, 'hybrid packed weights on eager CUDA require '
                                  'the Triton projection kernels')
        ok, refusal = admit_q35(Candidate(quantization='q8_0', precision='float16',
                                          kernels='torch', prefill='fla',
                                          graphs=True, max_batch=1))
        self.assertFalse(ok)
        self.assertEqual(refusal, 'FLA prefill requires the CUDA Triton backend')

    def test_numeric_bounds(self):
        for bad in (0, 9, 4.0):
            ok, refusal = admit_q35(Candidate(max_batch=bad))
            self.assertFalse(ok)
            self.assertIn('batch size must be 1..8', refusal)
        ok, refusal = admit_q35(Candidate(batch_wait_ms=11.0))
        self.assertFalse(ok)
        self.assertIn('0..10 ms', refusal)
        ok, refusal = admit_q35(Candidate(prefill_chunk_size=-1))
        self.assertFalse(ok)

    def test_experimental_optins_refuse_the_whole_search(self):
        with mock.patch.dict(os.environ, {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1'}):
            ok, refusal = admit_q35(Candidate(quantization='q8_0', precision='float16'))
            self.assertFalse(ok)
            self.assertIn('production policy surface', refusal)

    def test_admission_restores_the_ambient_environment(self):
        ambient = dict(os.environ)
        admit_q35(Candidate(quantization='q8_0', precision='float16'))
        admit_q35(Candidate(quantization='q4_0', precision='compensated'))  # refuses
        self.assertEqual(os.environ, ambient)


class DeclaredGridTests(unittest.TestCase):
    def test_manifest_owns_its_search(self):
        grid = declared_grid({'max_batch': [1, 6]}, 'q8_0')
        # Every undeclared axis collapses to the serve's own default.
        self.assertEqual(grid['precision'], ('float16',))
        self.assertEqual(grid['graphs'], (True,))
        self.assertEqual(grid['max_batch'], (1, 6))
        self.assertEqual(grid['quantization'], ('q8_0',))

    def test_declared_grid_validates(self):
        self.assertRaises(ValueError, declared_grid, {'vibes': [1]}, 'q8_0')
        self.assertRaises(ValueError, declared_grid, {'max_batch': []}, 'q8_0')
        self.assertRaises(ValueError, declared_grid, {'max_batch': [1]}, 'fp8')

    def test_grid_replaces_the_default_enumeration(self):
        grid = declared_grid({'max_batch': [1, 6]}, 'q8_0')
        admitted, refusals = enumerate_candidates(
            QWEN35_TEXT_CONFIG, accelerator='cuda', context=32768,
            has_mtp=False, grid=grid)
        self.assertEqual(sorted({c.max_batch for c in admitted}), [1, 6])
        self.assertTrue(all(c.precision == 'float16' for c in admitted))
        # A Q4 model on the production surface admits nothing.
        grid_q4 = declared_grid({}, 'q4_0')
        admitted_q4, refusals_q4 = enumerate_candidates(
            QWEN35_TEXT_CONFIG, accelerator='cuda', context=512,
            has_mtp=False, grid=grid_q4)
        self.assertEqual(admitted_q4, [])
        self.assertTrue(any('experimental hybrid Q4' in refusal
                            for _, refusal in refusals_q4))


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_roundtrips(self):
        manifest = cuda_manifest()
        parsed = parse_manifest(json.dumps(manifest))
        self.assertEqual(parsed, manifest)
        raw = json.dumps(manifest).encode()
        self.assertEqual(parse_manifest(raw), manifest)

    def test_rejects_known_bad_manifests(self):
        bad = [
            ({'schema_version': 2}, 'schema_version'),
            ({'schema_version': 1}, 'model'),
        ]
        manifest = cuda_manifest()
        def mutate(**changes):
            broken = dict(manifest)
            broken.update(changes)
            return json.dumps(broken)
        self.assertRaises(ValueError, parse_manifest, mutate(schema_version=2))
        self.assertRaises(ValueError, parse_manifest, mutate(hardware={'accelerator': 'tpu'}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(corpus={'prompt_sha256': 'z' * 63, 'prompt_tokens': 1}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(corpus={'prompt_sha256': 'A' * 64, 'prompt_tokens': 1}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(requests={'count': 6, 'max_tokens': 512, 'concurrency': 7}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(objective={'metric': 'vibes', 'direction': 'maximize'}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(objective={'metric': 'p95_first_output_seconds',
                                             'direction': 'maximize'}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(search={'rounds': 0, 'timeout_seconds': 100}))
        self.assertRaises(ValueError, parse_manifest,
                           mutate(search={'rounds': 1, 'timeout_seconds': 1801}))
        self.assertRaises(ValueError, parse_manifest, 'not json')
        self.assertRaises(ValueError, parse_manifest,
                          json.dumps(manifest) + ' ' + 'x' * 300 * 1024)
        with self.assertRaises(ValueError):
            parse_manifest(json.dumps(manifest).replace('600.0', '1e999'))
        self.assertEqual(bad[0][1], 'schema_version')

    def test_corpus_digest_matches_the_pinned_workload_convention(self):
        prompt = ('The quick brown fox jumps over the lazy dog while the '
                  'repository records every bounded qualification cell. ')
        self.assertEqual(corpus_digest(prompt),
                         __import__('hashlib').sha256(prompt.encode()).hexdigest())


class ObjectiveTests(unittest.TestCase):
    ROWS = None

    def _row(self, value, first=100.0, tokens=(512,), errors=0, reason=None):
        return Measurement(
            candidate=Candidate(max_batch=1), metric_value=value,
            first_output_seconds=[first] * 4, request_seconds=[first + 400] * 4,
            completion_tokens=list(tokens), errors=errors,
            unmeasurable_reason=reason)

    def test_rank_orders_by_the_declared_metric_and_gates(self):
        objective = cuda_manifest()['objective']
        slow_ok = self._row(10.0)
        fast_ok = self._row(20.0)
        fast_but_gate_failed = self._row(30.0, first=900.0)  # first-output gate
        wrong_tokens = self._row(40.0, tokens=(511,))
        errored = self._row(50.0, errors=2)
        unmeasured = self._row(None, reason='co-tenant drift')
        ranked, eliminated = rank([slow_ok, fast_ok, fast_but_gate_failed,
                                   wrong_tokens, errored, unmeasured], objective)
        self.assertEqual([row.metric_value for row in ranked], [20.0, 10.0])
        eliminated_values = {row.metric_value for row, _ in eliminated}
        self.assertEqual(eliminated_values, {30.0, 40.0, 50.0, None})
        reasons = {row.metric_value: reasons for row, reasons in eliminated}
        self.assertTrue(any('exceeded' in r for r in reasons[30.0]))
        self.assertTrue(any('completion tokens' in r for r in reasons[40.0]))
        self.assertTrue(any('errors' in r for r in reasons[50.0]))

    def test_minimize_direction_ranks_lowest_first(self):
        objective = {'metric': 'p95_first_output_seconds', 'direction': 'minimize',
                     'gates': {'zero_errors': True}}
        rows = [self._row(20.0), self._row(5.0), self._row(10.0)]
        ranked, eliminated = rank(rows, objective)
        self.assertEqual([row.metric_value for row in ranked], [5.0, 10.0, 20.0])
        self.assertEqual(eliminated, [])

    def test_gate_evaluation_flags_each_declared_gate(self):
        gates = {'max_first_output_seconds': 10, 'max_request_seconds': 20,
                 'exact_completion_tokens': 8, 'zero_errors': True}
        passing = Measurement(Candidate(), metric_value=1, first_output_seconds=[5],
                              request_seconds=[15], completion_tokens=[8, 8], errors=0)
        self.assertEqual(evaluate_gates(passing, gates), [])
        reasons = evaluate_gates(self._row(1, first=11), gates)
        self.assertTrue(any('first output' in r for r in reasons))
        row = Measurement(Candidate(), metric_value=1, first_output_seconds=[5],
                          request_seconds=[21], completion_tokens=[7], errors=1)
        reasons = evaluate_gates(row, gates)
        self.assertEqual(len(reasons), 3)  # request, tokens, errors; first output passes


class WinnerOutputTests(unittest.TestCase):
    def test_winner_record_is_serve_consumable_and_reviewable(self):
        manifest = cuda_manifest()
        candidate = Candidate(quantization='q8_0', precision='float16',
                              kernels='triton', graphs=True, max_batch=6,
                              prefill_chunk_size=1024)
        record = winner_record(manifest, candidate, 93.5)
        model = record['model']
        self.assertEqual(model, {'version': 1,
                                 'models': {'qwen36': {'path': '/models/Qwen3.6-35B-A3B',
                                                      'quantization': 'q8_0'}}})
        # The exact environment + flags a reviewer needs to reproduce.
        self.assertEqual(record['environment']['DAMENDAEMON_HYBRID_PRECISION'], 'float16')
        self.assertEqual(record['environment']['DAMENDAEMON_HYBRID_MTP_DROP'], '1')
        self.assertIn('--prefill-chunk-size', record['serve_flags'])
        self.assertEqual(record['metric_value'], 93.5)
        self.assertEqual(record['objective']['metric'],
                         'aggregate_output_tokens_per_second')

    def test_serve_manifest_matches_the_serve_model_config_shape(self):
        emitted = serve_manifest('qwen36', '/models/Qwen3.6-35B-A3B', 'q8_0')
        self.assertEqual(emitted['version'], 1)
        self.assertEqual(set(emitted['models']), {'qwen36'})
        # The harness manifests (e.g. split_gguf.save) use this exact shape.
        self.assertEqual(set(emitted['models']['qwen36']), {'path', 'quantization'})


if __name__ == '__main__':
    unittest.main()
