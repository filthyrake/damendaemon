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

from damendaemon.nccl_transport_evidence import (DeadlineWatchdog,
                                                 ConsoleTranscript,
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
        self.assertEqual(record['ib_selection_lines'], [CHANNEL_IB, IB_USING])
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
        self.assertEqual(record['socket_selection_lines'], [CHANNEL_SOCKET,
                                                            SOCKET_USING])

    def test_real_socket_cell_transcript_is_socket_despite_ib_fallback_lines(self):
        # Verbatim shape from the live 2026-10-27 cross-host socket cell:
        # NCCL 2.29.7 with zero RDMA hardware STILL prints 'NET/IB : Using
        # [RO]' (plugin fallback claim) beside 'No device found', while
        # every data channel runs via NET/Socket — channels win.
        record = parse_transport_evidence([
            "dhcp:1:1 [0] NCCL INFO NET/Socket : Using [0]enp6s18:10.30.100.3<0>",
            "dhcp:1:1 [0] NCCL INFO NET/IB : No device found.",
            "dhcp:1:1 [0] NCCL INFO NET/IB : Using [RO]; OOB enp6s18:10.30.100.3<0>",
            "dhcp:1:1 [0] NCCL INFO NET/Socket : GPU Direct RDMA Disabled for HCA 0 'enp6s18'",
            "dhcp:1:1 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[1] [receive] via NET/Socket/0",
            "dhcp:1:1 [0] NCCL INFO Channel 01/0 : 0[1] -> 1[0] [send] via NET/Socket/0",
        ])
        self.assertEqual(record['nccl_selected_transport'], 'socket')
        self.assertEqual(record['gpudirect_rdma'], 'not_applicable')

    def test_plugin_using_lines_classify_only_without_channel_lines(self):
        # No channel lines at all (transcript truncated before the grid):
        # fall back to the plugin 'Using' claims.
        record = parse_transport_evidence([SOCKET_USING,
                                           "NCCL INFO NET/IB : Using [RO]; OOB x"])
        self.assertEqual(record['nccl_selected_transport'], 'ib')
        record = parse_transport_evidence([SOCKET_USING])
        self.assertEqual(record['nccl_selected_transport'], 'socket')

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
        # Live-path sanity only: a list of strings, and matching whatever
        # this host actually exposes (a CI runner may legitimately have
        # RDMA hardware — one reported 'mlx5_0' on this test's first merge
        # attempt — so emptiness is NOT asserted; the fleet inventory
        # record lives in the evidence README, not here).
        devices = rdma_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(all(isinstance(name, str) for name in devices))

    def test_rdma_devices_with_injected_sysfs_root(self):
        # Controlled-root behavior: devices are listed, sorted, and the
        # missing-directory case is an honest empty list, not an error.
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(rdma_devices(sysfs_root=root), [])
            os.makedirs(os.path.join(root, 'class', 'infiniband', 'mlx5_1'))
            os.makedirs(os.path.join(root, 'class', 'infiniband', 'mlx5_0'))
            self.assertEqual(rdma_devices(sysfs_root=root),
                             ['mlx5_0', 'mlx5_1'])


class ConsoleTranscriptTests(unittest.TestCase):

    def test_redirect_captures_both_fds_and_restores_console(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'transcript.log')
            transcript = ConsoleTranscript(path)
            transcript.install()
            with self.assertRaises(RuntimeError):
                transcript.install()
            import sys
            # fd 1 is NCCL's logging channel (measured, #602); fd 2 carries
            # tracebacks. Both must land in the transcript.
            stdout_marker = b'stdout fd-level write (NCCL channel)\n'
            os.write(1, stdout_marker)
            print('python stdout line', flush=True)
            marker = b'fd-level write beyond sys.stderr\n'
            os.write(2, marker)
            print('python-level line', file=sys.stderr)
            lines = transcript.close()
            with self.assertRaises(RuntimeError):
                transcript.close()
            self.assertIsInstance(lines, list)
            self.assertTrue(any(stdout_marker.rstrip() in line.encode()
                                for line in lines))
            self.assertTrue(any('python stdout line' in line for line in lines))
            self.assertTrue(any(marker.rstrip() in line.encode()
                                for line in lines))
            self.assertTrue(any('python-level line' in line for line in lines))
            # Both descriptors are restored and functional for later writes.
            os.write(1, b'after-restore-1\n')
            os.write(2, b'after-restore-2\n')
            content = Path(path).read_bytes()
            self.assertNotIn(b'after-restore', content)

    def test_failed_install_leaves_stderr_untouched(self):
        with self.assertRaises(ValueError):
            ConsoleTranscript('')
        transcript = ConsoleTranscript('/nonexistent-dir/transcript.log')
        with self.assertRaises(OSError):
            transcript.install()
        with self.assertRaises(RuntimeError):
            transcript.close()


class DeadlineWatchdogTests(unittest.TestCase):

    def test_armed_watchdog_cancels_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = ConsoleTranscript(str(Path(directory) / 't.log'))
            transcript.install()
            try:
                watchdog = DeadlineWatchdog(60, 'test-cell')
                with watchdog:
                    pass  # cancelled on exit; nothing fires
                self.assertFalse(watchdog.fired)
            finally:
                transcript.close()

    def test_reuse_and_invalid_labels_are_refused(self):
        watchdog = DeadlineWatchdog(60, 'first')
        with watchdog:
            pass
        with self.assertRaises(RuntimeError):
            with watchdog:  # already used once
                pass
        with self.assertRaises(ValueError):
            DeadlineWatchdog(0, 'bad')
        with self.assertRaises(ValueError):
            DeadlineWatchdog(60, '')


if __name__ == '__main__':
    unittest.main()
