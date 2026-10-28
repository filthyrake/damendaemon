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

# Substring markers, matched case-sensitively as NCCL prints them.
GPUDIRECT_LINE_MARKER = 'GPU Direct RDMA'
GPUDIRECT_ENABLED_MARKERS = ('RDMA enabled', 'RDMA Enabled')
GPUDIRECT_UNAVAILABLE_MARKERS = ('RDMA not available',)
# Measured on the #602 socket cell (NCCL 2.29.7): the socket plugin prints
# "GPU Direct RDMA Disabled for HCA 0 '<iface>'" — an explicit refused
# verdict, distinct from the IB plugin's "not available" wording.
GPUDIRECT_DISABLED_MARKERS = ('RDMA Disabled',)


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
    the per-HCA detail stays inspectable).

    Transport classification is anchored on the per-channel data-path
    lines ('via NET/IB/<n>' / 'via NET/Socket/<n>') when any exist: the
    plugin 'Using' lines can appear for fallback/probe paths — measured on
    the #602 socket cell, NCCL 2.29.7 prints BOTH 'NET/IB : No device
    found.' AND 'NET/IB : Using [RO]; OOB ...' on a host with zero RDMA
    hardware while every channel runs 'via NET/Socket/0'. Plugin 'Using'
    lines are recorded and used only when no channel line exists. Verdicts:

    - gpudirect_rdma 'evidenced': NCCL reported an enabled-GPUDirect HCA.
    - 'not_available': NCCL ran its GPUDirect check and every HCA refused
      (or explicitly printed Disabled).
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
    channel_ib, channel_socket = [], []
    plugin_ib, plugin_socket = [], []
    gpudirect_enabled, gpudirect_unavailable = [], []
    for line in lines:
        if 'via NET/IB/' in line:
            channel_ib.append(line)
        if 'via NET/Socket/' in line:
            channel_socket.append(line)
        if 'NET/IB : Using' in line:
            plugin_ib.append(line)
        if 'NET/Socket : Using' in line:
            plugin_socket.append(line)
        if GPUDIRECT_LINE_MARKER in line:
            if any(marker in line for marker in GPUDIRECT_ENABLED_MARKERS):
                gpudirect_enabled.append(line)
            elif any(marker in line for marker in GPUDIRECT_UNAVAILABLE_MARKERS
                     + GPUDIRECT_DISABLED_MARKERS):
                gpudirect_unavailable.append(line)
    if channel_ib:
        transport = 'ib'
    elif channel_socket:
        transport = 'socket'
    elif plugin_ib:
        transport = 'ib'
    elif plugin_socket:
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
                ib_selection_lines=channel_ib + plugin_ib,
                socket_selection_lines=channel_socket + plugin_socket,
                gpudirect_lines=gpudirect_enabled + gpudirect_unavailable)


class ConsoleTranscript:
    """Redirect the process console (fd 1 and fd 2) into a transcript file.

    Measured on the fleet (torch 2.13.0+cu130 / NCCL 2.29.7, #602): NCCL's
    C library logs to STDOUT (fd 1), Python tracebacks to stderr (fd 2) —
    the earlier fd-2-only redirect captured neither NCCL line (reports read
    'unknown' transport with the full grid green). This is a DIRECT
    kernel-level redirect (dup2 of the transcript file onto fd 1 AND fd 2):
    every writer holding either descriptor — prints, tracebacks,
    faulthandler, NCCL, forked NCCL proxy processes — keeps writing
    unchanged, now into the file. No pump thread by design: the earlier tee
    (2026-10-24 WIP incident) produced 0-byte transcripts on Linux while
    console passthrough worked — the fork/NCCL-proxy interaction with the
    pump thread is the recorded suspect — and a dup2 redirect has no
    userspace moving parts to break. The launcher's own nohup redirection
    keeps an (empty) console log; the completed transcript is read back for
    the report at close.

    Re-entrant install/close is refused. A failed install leaves both
    descriptors untouched.
    """

    def __init__(self, path):
        if not isinstance(path, str) or not path:
            raise ValueError('transcript path required')
        self.path = path
        self._saved = None

    def install(self):
        if self._saved is not None:
            raise RuntimeError('console transcript already installed')
        file = open(self.path, 'wb')
        try:
            saved1 = os.dup(1)
            try:
                saved2 = os.dup(2)
                try:
                    os.dup2(file.fileno(), 1)
                    os.dup2(file.fileno(), 2)
                except BaseException:
                    os.close(saved2)
                    raise
            except BaseException:
                os.close(saved1)
                raise
        except BaseException:
            file.close()
            raise
        file.close()  # fds 1/2 own the description now; drop our handle
        self._saved = (saved1, saved2)

    def close(self):
        """Restore fd 1/2 and return the captured transcript lines."""
        if self._saved is None:
            raise RuntimeError('console transcript not installed')
        saved1, saved2 = self._saved
        self._saved = None
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)
        # Both descriptors are restored; the transcript is complete because
        # every console writer was redirected through the same description.
        with open(self.path, 'rb') as captured:
            return captured.read().decode('utf-8', errors='replace').splitlines()


class DeadlineWatchdog:
    """Process-fatal outer deadline for unbounded C-level waits (#602).

    torch's NCCL work.wait() and the collectives here have no per-call
    deadline the process-group timeout reliably covers: the 2026-10-24
    cross-host round_trip hang spun both ranks at 100% CPU for 77+
    minutes with the 30 s group timeout never firing (recorded in the
    #602 incident note — the same blocking-collective process-fatal gap
    docs/multi-gpu.md records for supervision). This watchdog arms a
    daemon thread per guarded region; at expiry it writes a labeled
    marker and every thread's traceback to fd 2 (which, under
    StderrTranscript, is the evidence transcript) and exits the process
    with status 70 — a recorded deadline kill instead of a silent hang.
    Incremental report writes mean the report on disk names the last
    COMPLETED cell; the transcript names the stuck one.

    Single-use; reuse, zero/negative budgets and empty labels are
    refused. Context-manager form cancels cleanly on exit.
    """

    EXIT_STATUS = 70

    def __init__(self, seconds, label):
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError('deadline seconds must be positive')
        if not isinstance(label, str) or not label:
            raise ValueError('deadline label required')
        self.seconds = seconds
        self.label = label
        self.fired = False
        self._armed = False
        self._done = False
        import threading
        self._cancel = threading.Event()
        self._thread = None

    def _fire(self):
        self.fired = True
        marker = (f'deadline-watchdog: {self.label} exceeded '
                  f'{self.seconds}s; dumping tracebacks and exiting '
                  f'{self.EXIT_STATUS}\n').encode()
        try:
            os.write(2, marker)
            import faulthandler
            import sys
            faulthandler.dump_traceback(file=sys.stderr)
        except BaseException:
            pass
        os._exit(self.EXIT_STATUS)

    def __enter__(self):
        if self._armed or self._done:
            raise RuntimeError('watchdog is single-use')
        self._armed = True
        import threading

        def countdown():
            if self._cancel.wait(self.seconds):
                return
            self._fire()

        self._thread = threading.Thread(target=countdown, daemon=True,
                                        name=f'deadline-{self.label}')
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        thread = self._thread
        self._cancel.set()
        if thread is not None:
            thread.join(5)
        self._armed = False
        self._done = True
        return False
