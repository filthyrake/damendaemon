import tempfile
import unittest
from pathlib import Path

import torch

from scripts.experiments import thermal_pacing
from damendaemon import thermal_pacing as runtime_pacing


def write_hwmon(root, directory, driver, values):
    target = Path(root)/directory
    target.mkdir(parents=True)
    (target/'name').write_text(driver)
    for index, value in enumerate(values, start=1):
        (target/f'temp{index}_input').write_text(str(int(value*1000)))
    return target


class ThermalPacingTest(unittest.TestCase):
    def test_read_sensors_scans_k10temp_and_amdgpu_only(self):
        with tempfile.TemporaryDirectory() as root:
            write_hwmon(root, 'hwmon1', 'k10temp', [37.5])
            write_hwmon(root, 'hwmon2', 'amdgpu', [80.0, 35.0])
            write_hwmon(root, 'hwmon3', 'iwlwifi', [40.0])
            readings = dict(thermal_pacing.read_sensors(root))
        self.assertEqual(readings, {'k10temp/temp1_input': 37.5, 'amdgpu/temp1_input': 80.0,
                                    'amdgpu/temp2_input': 35.0})

    def test_read_sensors_rejects_out_of_range(self):
        with tempfile.TemporaryDirectory() as root:
            write_hwmon(root, 'hwmon1', 'amdgpu', [160.0])
            with self.assertRaises(RuntimeError):
                thermal_pacing.read_sensors(root)

    def test_gate_waits_until_below_soft_stop_and_records(self):
        sequence = [[('amdgpu/temp1_input', 81.0)], [('amdgpu/temp1_input', 80.0)],
                    [('amdgpu/temp1_input', 79.0)]]
        sleeps = []
        report = {}
        pacer = thermal_pacing.ThermalPacer(
            report, soft_stop=80.0, poll_seconds=0.5,
            read=lambda: sequence.pop(0) if sequence else [('amdgpu/temp1_input', 79.0)],
            sleep=sleeps.append, clock=lambda: 0.0)
        waited = pacer.gate('event')
        self.assertEqual(waited, 1.0)
        self.assertEqual(sleeps, [0.5, 0.5])
        self.assertEqual(report['gates'], 1)
        self.assertEqual(report['waited_seconds'], 1.0)
        self.assertEqual(report['maxima'], {'amdgpu/temp1_input': 81.0})
        self.assertEqual(report['events'][0]['reason'], 'event')

    def test_gate_returns_immediately_when_cool(self):
        report = {}
        pacer = thermal_pacing.ThermalPacer(report, soft_stop=80.0,
                                            read=lambda: [('k10temp/temp1_input', 40.0)],
                                            sleep=lambda _: None, clock=lambda: 0.0)
        self.assertEqual(pacer.gate('quantize'), 0.0)
        self.assertEqual(report['gates'], 1)
        self.assertEqual(report['waited_seconds'], 0.0)
        self.assertEqual(report['events'], [])

    def test_gate_logs_only_waiting_events(self):
        report = {}
        pacer = thermal_pacing.ThermalPacer(
            report, soft_stop=80.0,
            read=lambda: [('k10temp/temp1_input', 60.0)],
            sleep=lambda _: None, clock=lambda: 0.0, event_log_limit=2)
        for _ in range(5):
            pacer.gate('event')
        self.assertEqual(report['events'], [])

    def test_resolve_device_finds_device_arguments(self):
        self.assertIsNone(runtime_pacing._resolve_device((), {}))
        self.assertIsNone(runtime_pacing._resolve_device((torch.float32,), {}))
        device = runtime_pacing._resolve_device((), {'device': 'cuda'})
        self.assertEqual(device.type, 'cuda')
        device = runtime_pacing._resolve_device((torch.device('cuda:0'),), {})
        self.assertEqual((device.type, device.index), ('cuda', 0))
        self.assertIsNone(runtime_pacing._resolve_device((0,), {}))
        self.assertIsNone(runtime_pacing._resolve_device(('cpu',), {}))

    def test_activate_gates_only_large_transfers_to_cuda_targets(self):
        report = {}
        reasons = []
        pacer = thermal_pacing.ThermalPacer(report, soft_stop=80.0,
                                            read=lambda: [('k10temp/temp1_input', 40.0)],
                                            sleep=lambda _: None, clock=lambda: 0.0)
        original_gate = pacer.gate
        pacer.gate = lambda reason: reasons.append(reason)
        deactivate = thermal_pacing.activate(pacer, min_transfer_bytes=64)
        real_resolve = runtime_pacing._resolve_device
        runtime_pacing._resolve_device = lambda a, k: torch.device('cuda')
        try:
            torch.zeros(4).to('cpu')
            self.assertEqual(reasons, [])
            torch.zeros(2*1024*1024).to(device='cpu')
            self.assertEqual(reasons, ['transfer'])
            moved = torch.zeros(4).to('cpu', non_blocking=False)
            self.assertEqual(moved.device.type, 'cpu')
        finally:
            runtime_pacing._resolve_device = real_resolve
            deactivate()
            pacer.gate = original_gate
        self.assertIs(torch.Tensor.to, runtime_pacing.torch.Tensor.to)


if __name__ == '__main__':
    unittest.main()
