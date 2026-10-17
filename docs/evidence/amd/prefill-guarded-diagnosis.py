import json, os, signal, subprocess, sys, time
from pathlib import Path
from damendaemon.rocm_policy import host_available
from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog
output=Path(sys.argv[1]); command=sys.argv[2:]
if output.exists(): raise SystemExit('refusing to overwrite evidence')
if host_available()<80*1024**3: raise SystemExit('need 80 GiB initial host headroom')
report=dict(command=command,host_available_before=host_available(),watchdog={})
started=time.monotonic()
with output.with_suffix('.stdout').open('x') as stdout, output.with_suffix('.stderr').open('x') as stderr:
    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
    def stop():
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    guard=HostMemoryWatchdog(stop,report['watchdog']).start()
    try:
        report['exit_code']=child.wait(timeout=900)
    except subprocess.TimeoutExpired:
        report['timed_out']=True;stop();report['exit_code']=child.wait()
    finally:
        if child.poll() is None: stop();child.wait()
        guard.close()
        report.update(seconds=time.monotonic()-started,host_available_after=host_available(),cleanup_complete=child.poll() is not None)
        output.write_text(json.dumps(report,indent=2)+'\n')
if report['exit_code'] or report['watchdog']['triggered']:raise SystemExit(1)
