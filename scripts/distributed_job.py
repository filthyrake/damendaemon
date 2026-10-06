"""POSIX test-job deadlines that also terminate owned torchrun workers.

Each invocation creates its own process group. Never discover or terminate
GPU processes by name: existing services on shared hosts are unrelated jobs.
Use torchrun --run-path: subprocess/module workers create separate sessions
and cannot be force-terminated by this process-group deadline.
Call from the main thread; the harness's SIGTERM handler is restored on exit.
"""
import os
import signal
import subprocess


def _signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _terminate(process, grace):
    _signal_group(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        # Even cleanup is bounded: a wedged driver or escaped descendant must
        # produce a reported cleanup failure instead of hanging this harness.
        return process.communicate(timeout=grace)


def run_job(command, *, timeout=180, check=True, termination_grace=5):
    if os.name != 'posix':raise RuntimeError('distributed test jobs require POSIX process groups')
    if timeout <= 0 or termination_grace <= 0:raise ValueError('positive job and cleanup deadlines required')
    process = None
    terminating = False
    termination_requested = False

    def terminate_handler(number, frame):
        nonlocal termination_requested
        termination_requested = True
        # Defer interruption until Popen has returned its ownership handle.
        if process is not None and not terminating:raise SystemExit(128 + number)

    previous = signal.signal(signal.SIGTERM, terminate_handler)
    try:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            if termination_requested:raise SystemExit(128 + signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as error:
            if process is None:raise
            terminating = True
            try:
                stdout, stderr = _terminate(process, termination_grace)
                if isinstance(error, subprocess.TimeoutExpired):
                    error.output, error.stderr = stdout, stderr
            except BaseException as cleanup_error:
                error.add_note(f'owned process group {process.pid} cleanup failed: {cleanup_error!r}')
                process.stdout.close()
                process.stderr.close()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:result.check_returncode()
        return result
    finally:
        signal.signal(signal.SIGTERM, previous)
