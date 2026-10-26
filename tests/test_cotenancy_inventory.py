import unittest
from unittest import mock

from damendaemon import cotenancy_inventory, fleet_preflight


class CotenancyInventoryTest(unittest.TestCase):
    def test_mig_reports_unsupported_on_consumer_cards(self):
        stdout = ('index, name, mig.mode.current, mig.mode.pending\n'
                  '0, NVIDIA GeForce RTX 3090, [N/A], [N/A]\n')
        def fake_run(command, capture_output, text, timeout):
            result = mock.Mock()
            result.returncode = 0
            result.stdout = stdout
            return result
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-smi'), \
             mock.patch.object(cotenancy_inventory.subprocess, 'run', side_effect=fake_run):
            report = cotenancy_inventory.mig_inventory()
        self.assertFalse(report['supported'])
        self.assertTrue(report['passed'])
        self.assertIn('A30/data-center path', report['note'])

    def test_mig_mixed_host_reports_the_capable_gpu_per_row(self):
        """Regression (#609 cell): a consumer card must not hide a MIG slice.

        The A30+A2 host caught the whole-output 'N/A' shortcut reporting
        MIG unsupported while a 1g.6gb slice was live on the A30.
        """
        stdout = ('index, name, mig.mode.current, mig.mode.pending\n'
                  '0, NVIDIA A30, Enabled, Enabled\n'
                  '1, NVIDIA A2, [N/A], [N/A]\n')
        listing = ('GPU 0: NVIDIA A30 (UUID: GPU-x)\n'
                   '  MIG 1g.6gb  Device 0: (UUID: MIG-y)\n')
        def fake_run(command, capture_output, text, timeout):
            result = mock.Mock()
            result.returncode = 0
            result.stdout = listing if '-L' in command else stdout
            return result
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-smi'), \
             mock.patch.object(cotenancy_inventory.subprocess, 'run', side_effect=fake_run):
            report = cotenancy_inventory.mig_inventory()
        self.assertTrue(report['passed'])
        self.assertTrue(report['supported'])
        self.assertEqual(len(report['capable_gpus']), 1)
        self.assertEqual(report['capable_gpus'][0]['name'], 'NVIDIA A30')
        self.assertTrue(any('MIG 1g.6gb' in line for line in report['instances']))

    def test_compute_mode_flags_exclusive(self):
        with mock.patch.object(cotenancy_inventory, 'nvidia_smi', return_value=[
                dict(index='0', name='RTX 3090', compute_mode='Exclusive_Process')]):
            report = cotenancy_inventory.compute_mode_inventory()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['blocking']), 1)

    def test_mps_reports_missing_control_binary(self):
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value=None), \
             mock.patch.object(cotenancy_inventory, 'nvidia_smi', return_value=[]):
            report = cotenancy_inventory.mps_inventory()
        self.assertIsNone(report['control_binary'])
        self.assertFalse(report['server_running'])

    def test_compute_mode_skips_honestly_without_nvidia_smi(self):
        with mock.patch.object(cotenancy_inventory, 'nvidia_smi', return_value=None):
            report = cotenancy_inventory.compute_mode_inventory()
        self.assertTrue(report['passed'])
        self.assertIn('not applicable', report['note'])

    def test_mps_detects_running_server(self):
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-cuda-mps-control'), \
             mock.patch.object(cotenancy_inventory, 'nvidia_smi', return_value=[
                 dict(pid='1234', process_name='nvidia-cuda-mps-server', used_memory='256 MiB')]):
            report = cotenancy_inventory.mps_inventory()
        self.assertTrue(report['server_running'])

    def test_compute_mode_parses_real_header_format(self):
        """Regression (#609 cell): the query must use --format=csv (header).

        The nvidia_smi helper parses the FIRST line as the column header;
        a noheader query eats the first GPU row and KeyErrors on every
        NVIDIA host (this shipped bug only surfaced in the live MIG cell —
        HOMER has no nvidia-smi, and the unit tests mocked nvidia_smi).
        """
        stdout = ('index, name, compute_mode\n'
                  '0, NVIDIA A30, Default\n'
                  '1, NVIDIA A2, Default\n')
        def fake_run(command, capture_output, text, timeout):
            result = mock.Mock()
            result.returncode = 0
            result.stdout = stdout
            return result
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-smi'), \
             mock.patch.object(cotenancy_inventory.subprocess, 'run', side_effect=fake_run):
            report = cotenancy_inventory.compute_mode_inventory()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['gpus']), 2)
        self.assertEqual(report['gpus'][0]['compute_mode'], 'Default')
        self.assertEqual(report['blocking'], [])

    def test_preflight_cotenancy_extension_composes(self):
        fake = mock.Mock()
        fake.__version__ = '2.6.0+cu124'
        fake.version.cuda = '12.4'
        fake.version.hip = None
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.cuda.get_device_properties.return_value = mock.Mock(name='RTX 3090', uuid='x')
        fake.cuda.mem_get_info.return_value = (2**30, 24 * 2**30)
        fake.cuda.nccl.version.return_value = (2, 21, 5)
        with mock.patch.dict('os.environ', {}, clear=False), \
             mock.patch.object(cotenancy_inventory.shutil, 'which', return_value=None), \
             mock.patch.object(cotenancy_inventory.subprocess, 'run',
                               side_effect=OSError('no nvidia-smi')):
            report = fleet_preflight.run_report_with_cotenancy(fake, peer=None)
        items = [item['item'] for item in report['items']]
        self.assertEqual(items, ['torch', 'gpus', 'mig', 'compute_mode', 'mps', 'topology'])
        # every item skips honestly when nvidia-smi is absent -> report passes
        self.assertTrue(report['passed'])
        self.assertEqual(
            [item['item'] for item in report['items'] if not item['passed']], [])


if __name__ == '__main__':
    unittest.main()
