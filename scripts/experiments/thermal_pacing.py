"""Thermally paced host preparation, residency fill and measured phases.

A software workaround for HOMER's die-edge thermal wall: hardware cannot
sustain unpaced ramp phases, so this module inserts idle cool-downs
between already-synchronous operations. Pacing never changes kernels,
arithmetic, or measured windows; an outer guard_homer run remains the
hard cutoff (fans fixed at maximum, --limit-celsius kill).

Two patch points mirror the two heat sources of a full-model run:
- torch.Tensor.to: large host-to-device transfers (the 16 GiB residency
  fill in HybridExecutor construction) wait until sensors are below the
  soft threshold.
- damendaemon.quantization.quantize: the CPU Q8 preparation waits per
  weight matrix (Q8 hybrid checkpoints have no prepared-weight cache).

Sensors mirror guard_homer's homer_cooling scan: every k10temp/amdgpu
hwmon input, in Celsius. Experiments call gate() between measured events
so back-to-back replay bursts stay bounded too.
"""
import time
from pathlib import Path

import torch


def read_sensors(hwmon_root='/sys/class/hwmon'):
    """Return [(label, celsius)] for every k10temp/amdgpu sensor, or raise."""
    readings = []
    for directory in sorted(Path(hwmon_root).glob('hwmon*')):
        driver = (directory/'name').read_text().strip()
        if driver not in ('k10temp', 'amdgpu'):
            continue
        for sensor in sorted(directory.glob('temp*_input')):
            value = int(sensor.read_text())/1000
            if not -20 <= value <= 150:
                raise RuntimeError(f'invalid temperature reading: {driver} {value}')
            readings.append((f'{driver}/{sensor.stem}', value))
    if not readings:
        raise RuntimeError('no k10temp/amdgpu sensors found')
    return readings


class ThermalPacer:
    """Hold operations until every tracked sensor is below the soft stop.

    Hysteresis is a single band: gate() sleeps while the hottest sensor is
    at or above `soft_stop` and returns once it falls below it. The band
    must stay several degrees under guard_homer's kill limit.
    """

    def __init__(self, report, *, soft_stop=80.0, poll_seconds=0.5,
                 read=read_sensors, sleep=time.sleep, clock=time.monotonic,
                 event_log_limit=256):
        if not 50 <= soft_stop <= 93:
            raise ValueError('soft stop must be within 50..93 Celsius')
        self.report = report
        self.soft_stop = soft_stop
        self.read, self.sleep, self.clock = read, sleep, clock
        self.event_log_limit = event_log_limit
        report.update(soft_stop_celsius=soft_stop, poll_seconds=poll_seconds,
                      gates=0, waited_seconds=0.0, maxima={}, events=[])

    def _record(self, sensors):
        for label, value in sensors:
            maxima = self.report['maxima']
            if value > maxima.get(label, float('-inf')):
                maxima[label] = value

    def gate(self, reason):
        """Wait until the hottest sensor is below the soft stop; return wait."""
        self.report['gates'] += 1
        started = self.clock()
        sensors = self.read()
        waited = 0.0
        while max(value for _, value in sensors) >= self.soft_stop:
            self._record(sensors)
            self.sleep(self.report['poll_seconds'])
            waited += self.report['poll_seconds']
            sensors = self.read()
        self._record(sensors)
        if waited:
            self.report['waited_seconds'] += waited
            if len(self.report['events']) < self.event_log_limit:
                self.report['events'].append(dict(reason=reason, waited_seconds=waited,
                    sensors={label: value for label, value in sensors}))
        return waited


def _resolve_device(args, kwargs):
    """Best-effort device argument of Tensor.to; None when not a device call."""
    device = kwargs.get('device')
    if device is None:
        for argument in args:
            if isinstance(argument, (torch.device, int)) or (
                    isinstance(argument, str) and argument.startswith(('cuda', 'hip'))):
                device = argument
                break
    if device is None or isinstance(device, int):
        return None
    return torch.device(device)


def activate(pacer, *, min_transfer_bytes=8*1024*1024):
    """Patch Tensor.to and quantization for paced ramp phases.

    Returns deactivate(); only large device transfers and per-matrix Q8
    quantization gate, so small ops and measured kernels pass through.
    """
    from damendaemon import quantization

    original_to = torch.Tensor.to
    original_quantize = quantization.quantize

    def paced_to(self, *args, **kwargs):
        device = _resolve_device(args, kwargs)
        if (device is not None and device.type == 'cuda'
                and self.device.type != 'cuda'
                and self.numel() * self.element_size() >= min_transfer_bytes):
            pacer.gate('transfer')
        return original_to(self, *args, **kwargs)

    def paced_quantize(weight):
        pacer.gate('quantize')
        return original_quantize(weight)

    torch.Tensor.to = paced_to
    quantization.quantize = paced_quantize
    pacer.report['min_transfer_bytes'] = min_transfer_bytes

    def deactivate():
        torch.Tensor.to = original_to
        quantization.quantize = original_quantize

    return deactivate
