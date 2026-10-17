import json, os, signal, subprocess, sys
from pathlib import Path
from damendaemon.rocm_policy import host_available
from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog
before=host_available()
if before < 32*1024**3: raise SystemExit('insufficient initial headroom')
report={'host_available_before':before,'watchdog':{}}
with open('../results/rocm-attention-memory.jsonl','w') as output, open('../results/rocm-attention-memory.log','w') as errors:
    child=subprocess.Popen([sys.executable,'-m','scripts.experiments.diagnose_rocm_attention_memory'],stdout=output,stderr=errors,start_new_session=True)
    def stop():
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    guard=HostMemoryWatchdog(stop,report['watchdog'],reserve_bytes=max(24*1024**3,before-4*1024**3)).start()
    try:
        report['exit_code']=child.wait(timeout=150)
    except subprocess.TimeoutExpired:
        report['timed_out']=True
        stop();report['exit_code']=child.wait()
    finally:
        if child.poll() is None: stop();child.wait()
        guard.close()
        report['host_available_after']=host_available()
        Path('../results/rocm-attention-memory.guard.json').write_text(json.dumps(report,indent=2)+'\n')
if report['exit_code'] or report['watchdog']['triggered']:raise SystemExit(1)
