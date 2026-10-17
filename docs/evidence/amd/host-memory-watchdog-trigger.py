import json, os, signal, subprocess, sys
from damendaemon.rocm_policy import host_available
from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog
before=host_available()
if before < 24*1024**3: raise SystemExit('insufficient host reserve for diagnostic')
process=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],start_new_session=True)
report={}
guard=HostMemoryWatchdog(lambda:os.killpg(process.pid,signal.SIGKILL),report,reserve_bytes=before+1024**3).start()
try:
    code=process.wait(timeout=3)
finally:
    guard.close()
    if process.poll() is None:
        process.kill();process.wait()
after=host_available()
print(json.dumps(dict(passed=code==-signal.SIGKILL and report['triggered'] and after>16*1024**3, child_exit_code=code, host_available_before=before,host_available_after=after,watchdog=report,limitation='Deliberately elevated reserve threshold; real /proc accounting and owned-child kill. No memory pressure induced and no GPU allocation.'),indent=2))
if code!=-signal.SIGKILL or not report['triggered']:raise SystemExit(1)
