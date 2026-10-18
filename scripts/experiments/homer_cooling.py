"""Fail-closed HOMER cooling checks, independent of Torch and GPU execution."""
from pathlib import Path
import time

from damendaemon.rocm_policy import host_available


class CoolingGuard:
    def __init__(self, report, *, hwmon_root='/sys/class/hwmon',
                 ec_root='/sys/class/ec_su_axb35', limit_celsius=80, clock=time.monotonic):
        if not 50 <= limit_celsius <= 93:
            raise ValueError('thermal cutoff must be within 50..93 Celsius')
        self.report = report
        self.root, self.ec_root, self.limit = Path(hwmon_root), Path(ec_root), limit_celsius
        self.clock, self.last_check = clock, None
        report.update(limit_celsius=limit_celsius, checks=0, fans=[],
                      maximum_observed_celsius=None, sensors=[], error=None)

    def check(self, *, startup=False):
        try:
            fans = []
            # Controller level5 is the user's maximum-fan prerequisite. Tach
            # floors detect stopped/slow fans; they are conservative operational
            # floors below the observed max-speed RPM, not a thermal guarantee.
            for name, floor in (('fan1', 3000), ('fan2', 3000), ('fan3', 1000)):
                directory = self.ec_root/name
                fan = dict(name=name, mode=(directory/'mode').read_text().strip(),
                           level=int((directory/'level').read_text()),
                           rpm=int((directory/'rpm').read_text()), minimum_rpm=floor)
                fans.append(fan)
            self.report['fans'] = fans
            if any(f['mode'] != 'fixed' or f['level'] != 5 or f['rpm'] < f['minimum_rpm'] for f in fans):
                raise RuntimeError('run fan-inference.sh on and wait for all three fans at maximum')
            readings = []
            drivers = set()
            for directory in sorted(self.root.glob('hwmon*')):
                driver = (directory/'name').read_text().strip()
                if driver not in ('k10temp', 'amdgpu'):
                    continue
                for sensor in sorted(directory.glob('temp*_input')):
                    value = int(sensor.read_text())/1000
                    if not -20 <= value <= 150:
                        raise RuntimeError('invalid temperature reading')
                    stem = sensor.name.removesuffix('_input')
                    label = directory/(stem+'_label')
                    critical = directory/(stem+'_crit')
                    cutoff = self.limit
                    if critical.exists():
                        exported = int(critical.read_text())/1000
                        if not 50 <= exported <= 150:
                            raise RuntimeError('invalid critical temperature')
                        cutoff = min(cutoff, exported-10)
                    readings.append(dict(driver=driver, label=label.read_text().strip() if label.exists() else stem,
                                         celsius=value, cutoff_celsius=cutoff))
                    drivers.add(driver)
            if drivers != {'k10temp', 'amdgpu'}:
                raise RuntimeError('both CPU and GPU temperature sensors are required')
            self.report['sensors'] = readings
            maximum = max(r['celsius'] for r in readings)
            previous = self.report['maximum_observed_celsius']
            self.report['maximum_observed_celsius'] = maximum if previous is None else max(previous, maximum)
            self.report['checks'] += 1
            if any(r['celsius'] >= r['cutoff_celsius']-(10 if startup else 0) for r in readings):
                raise RuntimeError('thermal limit reached' if not startup else 'host needs cooldown before experiment')
            self.last_check = self.clock()
        except Exception as error:
            self.report['error'] = str(error)
            raise

    def read_host_available(self):
        # Host-memory watchdog invokes this at 10Hz. Check fan readback/temperatures
        # once per second; a failed check triggers its owned-child kill callback.
        if self.last_check is None or self.clock()-self.last_check >= 1:
            self.check()
        return host_available()
