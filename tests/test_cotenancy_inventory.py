import unittest
from unittest import mock

from damendaemon import cotenancy_inventory, fleet_preflight


class CotenancyInventoryTest(unittest.TestCase):
    def test_mig_reports_unsupported_on_consumer_cards(self):
        def fake_run(command, capture_output, text, timeout):
            result = mock.Mock()
            result.returncode = 0
            result.stdout = 'NVIDIA GeForce RTX 3090, [N/A], [N/A]\n'
            return result
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-smi'), \
             mock.patch.object(cotenancy_inventory.subprocess, 'run', side_effect=fake_run):
            report = cotenancy_inventory.mig_inventory()
        self.assertFalse(report['supported'])
        self.assertTrue(report['passed'])
        self.assertIn('A30/data-center path', report['note'])

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

    def test_mps_detects_running_server(self):
        with mock.patch.object(cotenancy_inventory.shutil, 'which', return_value='/usr/bin/nvidia-cuda-mps-control'), \
             mock.patch.object(cotenancy_inventory, 'nvidia_smi', return_value=[
                 dict(pid='1234', process_name='nvidia-cuda-mps-server', used_memory='256 MiB')]):
            report = cotenancy_inventory.mps_inventory()
        self.assertTrue(report['server_running'])

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
             mock.patch.object(cotenancy_inventory.shutil, 'which', return_value=None):
            report = fleet_preflight.run_report_with_cotenancy(fake, peer=None)
        items = [item['item'] for item in report['items']]
        self.assertEqual(items, ['torch', 'gpus', 'mig', 'compute_mode', 'mps'])
        # with no nvidia-smi, compute_mode honestly fails -> the report fails
        self.assertFalse(report['passed'])
        self.assertFalse([i for i in report['items'] if i['item'] == 'compute_mode'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
