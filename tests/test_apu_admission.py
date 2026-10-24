import json
import unittest
from unittest import mock

from damendaemon import rocm_policy


class APUAdmissionTest(unittest.TestCase):
    def setUp(self):
        for name in ('DAMENDAEMON_HOST_RESERVE_GIB', 'DAMENDAEMON_FORCE_DISCRETE_ACCOUNTING',
                     'DAMENDAEMON_ROCM_MEMORY_GIB'):
            mock.patch.dict('os.environ', {}, clear=False).start()
        self.addCleanup(mock.patch.stopall)

    def test_apu_default_reserve_is_6_gib(self):
        self.assertEqual(rocm_policy.host_reserve_gib(100 * 2**30), 6.0)

    def test_apu_reserve_env_override(self):
        import os
        os.environ['DAMENDAEMON_HOST_RESERVE_GIB'] = '10'
        self.assertEqual(rocm_policy.host_reserve_gib(100 * 2**30), 10.0)

    def test_discrete_accounting_restores_16_gib(self):
        import os
        os.environ['DAMENDAEMON_FORCE_DISCRETE_ACCOUNTING'] = '1'
        self.assertEqual(rocm_policy.host_reserve_gib(100 * 2**30), 16.0)

    def make_fake_torch(self):
        fake = mock.Mock()
        fake.version.hip = '6.4'
        fake.cuda.mem_get_info.return_value = (111 * 2**30, 61 * 2**30)
        fake.cuda.memory_allocated.return_value = 0
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        properties = mock.Mock()
        properties.gcnArchName = 'gfx1151'
        fake.cuda.get_device_properties.return_value = properties
        return fake

    def test_reserve_values_bound_the_admission_term(self):
        # 52.74 GiB required (measured #68 C1) must be admitted under the APU
        # reserve when host availability allows, and refused under discrete.
        required = int(52.74 * 2**30)
        host_available = 61 * 2**30
        fake = self.make_fake_torch()
        with mock.patch.dict('os.environ', {'ROCR_VISIBLE_DEVICES': '0',
                                            'HIP_VISIBLE_DEVICES': '0',
                                            'CUDA_VISIBLE_DEVICES': ''}), \
             mock.patch.object(rocm_policy, 'host_available', return_value=host_available), \
             mock.patch.object(rocm_policy, 'memory_limit', return_value=56 * 2**30):
            self.assertEqual(rocm_policy.available_memory(fake), 55 * 2**30)
            with mock.patch.dict('os.environ', {'DAMENDAEMON_FORCE_DISCRETE_ACCOUNTING': '1'}):
                self.assertEqual(rocm_policy.available_memory(fake), 45 * 2**30)
            self.assertGreater(rocm_policy.available_memory(fake), required)
            with mock.patch.dict('os.environ', {'DAMENDAEMON_FORCE_DISCRETE_ACCOUNTING': '1'}):
                self.assertLess(rocm_policy.available_memory(fake), required)

    def test_admission_report_written_when_requested(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.json') as handle:
            report_path = handle.name
        fake = self.make_fake_torch()
        with mock.patch.dict('os.environ', {'DAMENDAEMON_ADMISSION_REPORT': report_path,
                                            'ROCR_VISIBLE_DEVICES': '0',
                                            'HIP_VISIBLE_DEVICES': '0'}), \
             mock.patch.object(rocm_policy, 'host_available', return_value=61 * 2**30), \
             mock.patch.object(rocm_policy, 'memory_limit', return_value=56 * 2**30):
            rocm_policy.available_memory(fake)
            components = json.loads(open(report_path).read())
        self.assertEqual(set(components), {'free', 'total', 'limit_minus_allocated',
                                           'host_minus_reserve', 'allocated'})


if __name__ == '__main__':
    unittest.main()
