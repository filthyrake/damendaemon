"""NCCL transport-evidence parser and stderr-capture tests (#602).

The transcript shapes below are the REAL format strings of the NCCL build
the fleet runs (libnccl.so 2.29.7, extracted from labnode3's venv), so the
parser is pinned to what the actual library prints — not to guesses. These
tests qualify the PARSING and the fd-2 tee, never RDMA or GPUDirect RDMA
themselves: no RDMA endpoint exists in this fleet yet and none is claimed.
"""
import os
import tempfile
import unittest
from pathlib import Path

from damendaemon.nccl_transport_evidence import (StderrTranscript,
                                                 parse_transport_evidence,
                                                 rdma_devices)

# Exact libnccl.so 2.29.7 format-string renderings.
IB_USING = "NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE v2 ; OOB enp6s18:ens1f1"
SOCKET_USING = "NCCL INFO NET/Socket : Using [0]eno1"
CHANNEL_IB = "NCCL INFO Channel 00/02 : 0[0] -> 1[1] [receive] via NET/IB/0"
CHANNEL_SOCKET = "NCCL INFO Channel 00/02 : 0[0] -> 1[1] [receive] via NET/Socket/0"
GDR_ENABLED = "NCCL INFO NET/IB : GPU Direct RDMA enabled for HCA 0 'mlx5_0'"
GDR_UNAVAILABLE = "NCCL INFO NET/IB : GPU Direct RDMA not available for HCA 0 'mlx5_0'"
GDR_ENABLED_NEW = "NCCL INFO NET/IB : GPU Direct RDMA Enabled for mlx5_0"


class ParseTransportEvidenceTests(unittest.TestCase):

    def test_ib_selection_with_enabled_gpudirect_is_evidenced(self):
        record = parse_transport_evidence(
            [IB_USING, GDR_ENABLED, CHANNEL_IB, 'NCCL INFO Init COMPLETE'])
        self.assertEqual(record['nccl_selected_transport'], 'ib')
        self.assertEqual(record['gpudirect_rdma'], 'evidenced')
        self.assertEqual(record['ib_selection_lines'], [IB_USING, CHANNEL_IB])
        self.assertIn(GDR_ENABLED, record['gpudirect_lines'])

    def test_ib_selection_with_refused_gpudirect_is_not_available(self):
        record = parse_transport_evidence([IB_USING, GDR_UNAVAILABLE])
        self.assertEqual(record['nccl_selected_transport'], 'ib')
        self.assertEqual(record['gpudirect_rdma'], 'not_available')
        self.assertEqual(record['gpudirect_lines'], [GDR_UNAVAILABLE])

    def test_per_hca_verdicts_are_all_retained(self):
        other = "NCCL INFO NET/IB : GPU Direct RDMA not available for HCA 1 'mlx5_1'"
        record = parse_transport_evidence([IB_USING, GDR_ENABLED, other])
        self.assertEqual(record['gpudirect_rdma'], 'evidenced')
        self.assertEqual(len(record['gpudirect_lines']), 2)

    def test_ib_without_gpudirect_line_is_unknown_never_guessed(self):
        record = parse_transport_evidence([IB_USING, CHANNEL_IB])
        self.assertEqual(record['nccl_selected_transport'], 'ib')
        self.assertEqual(record['gpudirect_rdma'], 'unknown')
        self.assertEqual(record['gpudirect_lines'], [])

    def test_newer_enabled_wording_is_recognized(self):
        record = parse_transport_evidence([IB_USING, GDR_ENABLED_NEW])
        self.assertEqual(record['gpudirect_rdma'], 'evidenced')

    def test_socket_selection_makes_gpudirect_not_applicable(self):
        record = parse_transport_evidence([SOCKET_USING, CHANNEL_SOCKET])
        self.assertEqual(record['nccl_selected_transport'], 'socket')
        self.assertEqual(record['gpudirect_rdma'], 'not_applicable')
        self.assertEqual(record['socket_selection_lines'], [SOCKET_USING,
                                                            CHANNEL_SOCKET])

    def test_empty_or_unrelated_transcript_is_unknown(self):
        for lines in ([], ['NCCL INFO Init COMPLETE'],
                      ['warning: something unrelated']):
            record = parse_transport_evidence(lines)
            self.assertEqual(record['nccl_selected_transport'], 'unknown')
            self.assertEqual(record['gpudirect_rdma'], 'unknown')

    def test_invalid_transcripts_are_rejected(self):
        for bad in ('line', [42], [None], [b'bytes'], ['x' * 5000],
                    [['nested']], ['nul\0byte']):
            with self.assertRaises(ValueError):
                parse_transport_evidence(bad)
        with self.assertRaises(ValueError):
            parse_transport_evidence(['ok'] * 100001)

    def test_rdma_devices_reflects_this_host(self):
        devices = rdma_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(all(isinstance(name, str) for name in devices))
        # No RDMA hardware exists in this fleet today; an empty inventory is
        # the expected honest state, and the probe's RDMA gate relies on it.
        self.assertEqual(devices, [])


class StderrTranscriptTests(unittest.TestCase):

    def test_tee_captures_fd2_writes_and_restores_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'transcript.log')
            transcript = StderrTranscript(path)
            transcript.install()
            with self.assertRaises(RuntimeError):
                transcript.install()
            marker = b'fd-level write beyond sys.stderr\n'
            os.write(2, marker)               # how NCCL logs: the raw fd
            import sys
            print('python-level line', file=sys.stderr)
            stalled = transcript.close()
            with self.assertRaises(RuntimeError):
                transcript.close()
            self.assertFalse(stalled)
            content = Path(path).read_bytes()
            self.assertIn(marker, content)
            self.assertIn(b'python-level line', content)
            # fd 2 is restored and functional for later writes.
            os.write(2, b'after-restore\n')
            self.assertNotIn(b'after-restore', content)

    def test_failed_install_leaves_stderr_untouched(self):
        with self.assertRaises(ValueError):
            StderrTranscript('')
        transcript = StderrTranscript('/nonexistent-dir/transcript.log')
        with self.assertRaises(OSError):
            transcript.install()
        with self.assertRaises(RuntimeError):
            transcript.close()


if __name__ == '__main__':
    unittest.main()
