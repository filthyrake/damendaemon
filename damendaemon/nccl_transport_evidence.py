"""NCCL transport and GPUDirect RDMA evidence from real debug transcripts (#602).

This module PARSES evidence; it never fabricates it. The recognized line
shapes are grounded in the format strings of the NCCL build the fleet
actually runs (libnccl.so 2.29.7, strings extracted from labnode3's venv):
'NET/IB : Using%s %s; OOB %s:%s', 'NET/Socket : Using%s',
'Channel %02d/%d : ... via NET/%s/%d', 'NET/%s : GPU Direct RDMA %s for
HCA %d '<hca>'' and the newer 'GPU Direct RDMA Enabled for %s'. What the
INFO-level lines prove is NCCL's OWN transport selection and per-HCA
GPUDirect capability check — not per-transfer DMA-bypass confirmation, and
not a throughput claim. RDMA and GPUDirect RDMA remain unqualified until
a real RDMA pair runs them (#602 is inventory-gated).

No transport is selected or forced here; no network is touched.
"""
import os
import re
import threading

# Substring markers, matched case-sensitively as NCCL prints them.
IB_SELECTION_MARKERS = ('NET/IB : Using', 'via NET/IB/')
SOCKET_SELECTION_MARKERS = ('NET/Socket : Using', 'via NET/Socket/')
GPUDIRECT_LINE_MARKER = 'GPU Direct RDMA'
GPUDIRECT_ENABLED_MARKERS = ('RDMA enabled', 'RDMA Enabled')
GPUDIRECT_UNAVAILABLE_MARKERS = ('RDMA not available',)


def rdma_devices():
    """Names of RDMA devices exposed to this worker; empty means none."""
    try:
        entries = os.listdir('/sys/class/infiniband')
    except OSError:
        return []
    return sorted(name for name in entries if name)


def parse_transport_evidence(lines):
    """Classify one captured NCCL debug transcript.

    Returns a bounded, JSON-serializable record of what the transcript
    itself shows: which network transport NCCL selected, and every
    GPUDirect RDMA verdict line it printed (per HCA; multiple HCAs may
    legitimately disagree — all matched lines are retained verbatim so
    the per-HCA detail stays inspectable). Verdicts:

    - gpudirect_rdma 'evidenced': NCCL reported an enabled-GPUDirect HCA.
    - 'not_available': NCCL ran its GPUDirect check and every HCA refused.
    - 'unknown' with an IB transport: the check line never appeared in
      this transcript (older/newer NCCL wording, or different SUBSYS) —
      absence is recorded as absence, never as failure or success.
    - 'not_applicable': NCCL selected the socket transport.

    An RDMA selection with no GPUDirect evidence is NOT GPUDirect RDMA
    qualification; 'evidenced' is NCCL's capability check, not a bypass
    confirmation for any executed transfer.
    """
    if (not isinstance(lines, (list, tuple))
            or len(lines) > 100000
            or any(not isinstance(line, str) or '\0' in line or len(line) > 4096
                   for line in lines)):
        raise ValueError('invalid NCCL transcript lines')
    selected_ib, selected_socket = [], []
    gpudirect_enabled, gpudirect_unavailable = [], []
    for line in lines:
        if any(marker in line for marker in IB_SELECTION_MARKERS):
            selected_ib.append(line)
        if any(marker in line for marker in SOCKET_SELECTION_MARKERS):
            selected_socket.append(line)
        if GPUDIRECT_LINE_MARKER in line:
            if any(marker in line for marker in GPUDIRECT_ENABLED_MARKERS):
                gpudirect_enabled.append(line)
            elif any(marker in line for marker in GPUDIRECT_UNAVAILABLE_MARKERS):
                gpudirect_unavailable.append(line)
    if selected_ib:
        transport = 'ib'
    elif selected_socket:
        transport = 'socket'
    else:
        transport = 'unknown'
    if transport == 'socket':
        gpudirect = 'not_applicable'
    elif gpudirect_enabled:
        gpudirect = 'evidenced'
    elif gpudirect_unavailable:
        gpudirect = 'not_available'
    else:
        gpudirect = 'unknown'
    return dict(nccl_selected_transport=transport, gpudirect_rdma=gpudirect,
                ib_selection_lines=selected_ib, socket_selection_lines=selected_socket,
                gpudirect_lines=gpudirect_enabled + gpudirect_unavailable)


class StderrTranscript:
    """Tee the process's real stderr (fd 2) into a transcript file.

    NCCL's C library logs to file descriptor 2, below Python's sys.stderr,
    so capturing its debug output needs the descriptor itself. Everything
    already holding fd 2 (Python tracebacks, faulthandler, NCCL) keeps
    printing to the console unchanged while the transcript accumulates.
    Re-entrant install/close is refused; close() restores fd 2 first, then
    the pump thread drains to EOF on its own. A failed install leaves
    stderr untouched.
    """

    def __init__(self, path):
        if not isinstance(path, str) or not path:
            raise ValueError('transcript path required')
        self.path = path
        self._saved = None
        self._file = None
        self._reader = None
        self._thread = None

    def install(self):
        if self._saved is not None:
            raise RuntimeError('stderr transcript already installed')
        file = open(self.path, 'wb', buffering=0)
        try:
            reader, writer = os.pipe()
            try:
                # fd 2 must stay BLOCKING: NCCL and Python write to it
                # synchronously, and a non-blocking pipe would error or
                # drop output whenever the pump thread falls behind.
                saved = os.dup(2)
                try:
                    os.dup2(writer, 2)
                except BaseException:
                    os.close(saved)
                    raise
            except BaseException:
                os.close(reader)
                os.close(writer)
                raise
        except BaseException:
            file.close()
            raise
        os.close(writer)
        self._file, self._reader, self._saved = file, reader, saved

        def pump():
            while True:
                try:
                    chunk = os.read(reader, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                file.write(chunk)
                try:
                    os.write(saved, chunk)
                except OSError:
                    pass

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    def close(self):
        """Restore fd 2 and finish the transcript; waits for the pump."""
        if self._saved is None:
            raise RuntimeError('stderr transcript not installed')
        saved = self._saved
        reader = self._reader
        file = self._file
        thread = self._thread
        self._saved = self._reader = self._file = self._thread = None
        os.dup2(saved, 2)
        os.close(saved)
        os.close(reader)  # no writer remains: pump drains to EOF and exits
        thread.join(30)
        file.close()
        return thread.is_alive()  # True means the pump never drained
