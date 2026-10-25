"""#604 topology-aware planning and the read-only topology inventory."""
import unittest
from unittest import mock

from damendaemon import topology_inventory
from damendaemon.hybrid_layout import choose_stage_splits
from tests import test_hybrid_stage


class TopologyPlanningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model

    def choose(self, budgets, **kwargs):
        return choose_stage_splits(self.model.weights, self.model.config, 32, 2, 4, budgets, True, **kwargs)

    def test_no_topology_matches_legacy_output(self):
        legacy = self.choose([2**30, 2**30])
        self.assertIn('topology_penalty_total', legacy)
        self.assertEqual(legacy['topology_penalty_total'], 0.0)
        self.assertIsNone(legacy['boundary_links'])

    def test_penalize_discloses_links_and_penalty(self):
        result = self.choose([2**30, 2**30, 2**30],
                             boundary_topology=[dict(peer_available=True, cross_numa=False),
                                                dict(peer_available=False, cross_numa=True)])
        self.assertTrue(result['storage_plan_found'])
        self.assertEqual(result['topology_penalty_total'], 1/8 + 1/16)
        self.assertEqual(result['boundary_links'],
                         [dict(ranks=[0, 1], peer_available=True, cross_numa=False),
                          dict(ranks=[1, 2], peer_available=False, cross_numa=True)])

    def test_peer_available_link_carries_no_penalty(self):
        clean = self.choose([2**30, 2**30],
                            boundary_topology=[dict(peer_available=True, cross_numa=False)])
        self.assertEqual(clean['topology_penalty_total'], 0.0)

    def test_exclude_refuses_non_p2p_ordering(self):
        result = self.choose([2**30, 2**30],
                             boundary_topology=[dict(peer_available=False, cross_numa=False)],
                             topology_policy='exclude')
        self.assertFalse(result['storage_plan_found'])
        self.assertIn('rank 0 and rank 1', result['reason'])

    def test_exclude_allows_p2p_ordering(self):
        result = self.choose([2**30, 2**30],
                             boundary_topology=[dict(peer_available=True, cross_numa=True)],
                             topology_policy='exclude')
        self.assertTrue(result['storage_plan_found'])

    def test_invalid_topology_arguments_rejected(self):
        with self.assertRaises(ValueError):
            self.choose([2**30], boundary_topology=[dict(peer_available=True)])  # length mismatch
        with self.assertRaises(ValueError):
            self.choose([2**30, 2**30], boundary_topology=[dict(peer_available=True)],
                        topology_policy='partition')


class TopologyInventoryTests(unittest.TestCase):
    def test_numa_node_minus_one_becomes_unknown(self):
        reader = lambda path: '-1\n'
        self.assertIsNone(topology_inventory.device_numa_and_affinity(
            '0000:01:00.0', reader=reader)['numa_node'])

    def test_numa_node_and_cpulist_parsed(self):
        def reader(path):
            return '0\n' if path.endswith('numa_node') else '0-7\n'
        info = topology_inventory.device_numa_and_affinity('0000:01:00.0', reader=reader)
        self.assertEqual(info['numa_node'], 0)
        self.assertEqual(info['local_cpulist'], '0-7')

    def test_missing_bus_id_reports_unknown_placement(self):
        info = topology_inventory.device_numa_and_affinity(None)
        self.assertIsNone(info['numa_node'])
        self.assertIn('unavailable', info['note'])

    def test_empty_report_when_no_cuda(self):
        torch = mock.Mock()
        torch.cuda.is_available.return_value = False
        torch.cuda.device_count.return_value = 0
        with mock.patch.object(topology_inventory, '_run', return_value=None), \
             mock.patch.object(topology_inventory, 'nic_numa_placement', return_value=[]):
            report = topology_inventory.run_topology_report(torch)
        self.assertTrue(report['passed'])
        self.assertEqual(report['devices'], [])
        self.assertEqual(report['peer_matrix'], [])
        self.assertIn('empty case', report['note'])

    def test_virtualization_markers_detected(self):
        def reader(path):
            if path.endswith('sys_vendor'):
                return 'QEMU\n'
            raise OSError
        markers = topology_inventory.virtualization_markers(reader=reader)
        self.assertTrue(markers['virtualized'])
        self.assertIn('ACS/IOMMU', markers['note'])

    def test_peer_matrix_records_errors_honestly(self):
        torch = mock.Mock()
        torch.cuda.can_device_access_peer.side_effect = RuntimeError('no peer')
        matrix = topology_inventory.peer_matrix(torch, 2)
        self.assertEqual(len(matrix), 2)
        self.assertFalse(matrix[0]['peer_available'])
        self.assertIn('error', matrix[0])


if __name__ == '__main__':
    unittest.main()
