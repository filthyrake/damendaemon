"""#731 peer-split policy contracts — all injected, no host environment."""
import unittest

from damendaemon.peer_split_policy import (
    DEFAULT_MIN_FREE_FRACTION, mutually_p2p_pairs, peer_split_decision,
    peer_split_enabled)

GIB = 2**30


def inventory(devices, pairs, capable=True):
    """A topology_inventory-shaped dict for the given devices/pairs."""
    matrix = []
    for a in range(len(devices)):
        for b in range(len(devices)):
            if a == b:
                continue
            on = capable and tuple(sorted((a, b))) in pairs
            matrix.append(dict(source=a, destination=b, peer_available=on))
    return dict(devices=[dict(index=i, total_memory_bytes=total)
                         for i, total in enumerate(devices)],
                peer_matrix=matrix)


THIRTY_NINETY_X2 = inventory([24 * GIB, 24 * GIB], [(0, 1)])
EIGHT_LAYERS = {layer: 5_457_051_648 for layer in range(58, 66)}  # #730's per-layer banks
# #730 measured working set: peak reserved 24,891,097,088 B − 21,828,206,592 B
# banks per GPU = 3,062,890,496 B (fleet chain fp8-oracle shape, tokens 8)
MEASURED_WORKING_SET = 3_062_890_496


class TriStateTests(unittest.TestCase):
    def test_default_off(self):
        import os
        saved = os.environ.pop('DAMENDAEMON_PEER_RESIDENT_SPLIT', None)
        try:
            self.assertFalse(peer_split_enabled())
        finally:
            if saved is not None:
                os.environ['DAMENDAEMON_PEER_RESIDENT_SPLIT'] = saved

    def test_enable_values(self):
        for value in ('', '1', 'true', 'on'):
            self.assertTrue(peer_split_enabled(value))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            peer_split_enabled('maybe')


class PairExtractionTests(unittest.TestCase):
    def test_mutual_only(self):
        # one-directional capability must NOT yield a pair
        rows = [dict(source=0, destination=1, peer_available=True),
                dict(source=1, destination=0, peer_available=False)]
        pairs = mutually_p2p_pairs(dict(devices=[dict(index=0, total_memory_bytes=1),
                                                 dict(index=1, total_memory_bytes=1)],
                                        peer_matrix=rows))
        self.assertEqual(pairs, [])

    def test_aggregate_ordering(self):
        inv = inventory([24 * GIB, 24 * GIB, 80 * GIB], [(0, 1), (1, 2)])
        pairs = mutually_p2p_pairs(inv)
        self.assertEqual(pairs[0][:2], (1, 2))          # biggest aggregate first
        self.assertEqual(pairs[0][2], 24 * GIB + 80 * GIB)

    def test_empty_inventory(self):
        self.assertEqual(mutually_p2p_pairs(dict(devices=[], peer_matrix=[])), [])


class DecisionTests(unittest.TestCase):
    def test_opt_in_required(self):
        decision: dict = peer_split_decision(EIGHT_LAYERS, THIRTY_NINETY_X2,
                                       working_set_bytes=MEASURED_WORKING_SET,
                                       enabled=False)
        self.assertEqual(decision['mode'], 'host_resident_stream')
        self.assertFalse(decision['opt_in'])
        self.assertIn('not enabled', decision['reason'])

    def test_split_selected_for_measured_range(self):
        decision: dict = peer_split_decision(EIGHT_LAYERS, THIRTY_NINETY_X2,
                                       working_set_bytes=MEASURED_WORKING_SET,
                                       enabled=True)
        self.assertEqual(decision['mode'], 'peer_resident_split')
        self.assertEqual(decision['pair'], (0, 1))
        self.assertFalse(decision['single_gpu_feasible'])
        # the arithmetic is the #730 shape: half the banks + working set
        self.assertEqual(decision['arithmetic']['per_gpu_banks_bytes'],
                         (8 * 5_457_051_648 + 1) // 2)
        peak = decision['arithmetic']['projected_peak_bytes']
        self.assertEqual(peak, (8 * 5_457_051_648 + 1) // 2 + MEASURED_WORKING_SET)
        # projected free fraction must clear the default threshold on both GPUs
        for device in ('0', '1'):
            self.assertGreaterEqual(
                decision['arithmetic']['projected_free_fraction'][device],
                DEFAULT_MIN_FREE_FRACTION)

    def test_nine_layers_refused_on_headroom(self):
        # a 9th layer of the same size class breaks the measured envelope
        layer_bytes = dict(EIGHT_LAYERS)
        layer_bytes[66] = 5_457_051_648
        decision: dict = peer_split_decision(layer_bytes, THIRTY_NINETY_X2,
                                       working_set_bytes=MEASURED_WORKING_SET,
                                       enabled=True)
        self.assertEqual(decision['mode'], 'host_resident_stream')
        self.assertIn('projected peak', decision['reason'])
        # the refusal still records the arithmetic it refused on
        self.assertLess(decision['arithmetic']['projected_free_fraction']['0'],
                        DEFAULT_MIN_FREE_FRACTION)

    def test_small_stage_needs_no_split(self):
        small = {layer: GIB for layer in range(4)}
        decision: dict = peer_split_decision(small, THIRTY_NINETY_X2, enabled=True)
        self.assertEqual(decision['mode'], 'host_resident_stream')
        self.assertTrue(decision['single_gpu_feasible'])
        self.assertIn('single device', decision['reason'])

    def test_no_p2p_pair_means_stream(self):
        inv = inventory([24 * GIB, 24 * GIB], [])
        decision: dict = peer_split_decision(EIGHT_LAYERS, inv, enabled=True)
        self.assertEqual(decision['mode'], 'host_resident_stream')
        self.assertIn('no mutually P2P-capable pair', decision['reason'])


if __name__ == '__main__':
    unittest.main()
