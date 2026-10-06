"""Harmless owned-child trip using real fan/temperature readings, without heat."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.experiments.homer_cooling import CoolingGuard
from scripts.experiments.host_memory_watchdog import HostMemoryWatchdog

report={'cooling':{},'watchdog':{},'limitation':'Test-only cutoff below idle temperature; no GPU work or induced heat. Normal launcher retains80C cutoff.'}
cooling=CoolingGuard(report['cooling'])
cooling.check(startup=True)
# Deliberately lower the internal test cutoff after normal preflight; ordinary
# launcher configuration does not permit this below-idle test threshold.
cooling.limit=min(r['celsius'] for r in report['cooling']['sensors'])-5
report['test_cutoff_override_celsius']=cooling.limit
cooling.last_check=None
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
watchdog=HostMemoryWatchdog(child.kill,report['watchdog'],read=cooling.read_host_available).start()
try:
    report['exit_code']=child.wait(timeout=5)
    report['passed']=report['exit_code']!=0 and report['watchdog']['triggered'] and 'thermal limit' in report['watchdog']['error']
finally:
    if child.poll() is None:child.kill();child.wait()
    watchdog.close()
    report['cleanup_complete']=child.poll() is not None
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
if not report['passed']:raise SystemExit(1)
