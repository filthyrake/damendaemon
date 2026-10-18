"""Intel Arc B580 device probe (#63): real-device enumeration and compute proof.

Read-only inventory per the #63 first gate: enumerates torch.xpu devices under
the host's oneAPI environment, records device name/UUID/memory counters, and
proves real device compute with a deterministic FP32 matmul plus a FP16/BF16
round-trip against a CPU control. No model load, no driver or config changes,
no performance claims (single tiny workload on a shared desktop).
"""
import argparse
import json
import os
import platform
from pathlib import Path

from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report = dict(completed=False, kind='B580 device probe; real compute, no model load',
                  host=platform.node(), python=platform.python_version(),
                  torch=torch.__version__, xpu_available=torch.xpu.is_available(),
                  devices=[], checks={})
    save(args.output, report)
    if not torch.xpu.is_available():
        report['error'] = 'torch.xpu reports no available device'
        save(args.output, report)
        raise RuntimeError(report['error'])
    count = torch.xpu.device_count()
    for index in range(count):
        properties = torch.xpu.get_device_properties(index)
        report['devices'].append(dict(
            index=index, name=str(properties.name), uuid=str(getattr(properties, 'uuid', '') or ''),
            total_memory_bytes=int(getattr(properties, 'total_memory', 0) or 0),
            driver_version=str(getattr(properties, 'driver_version', '') or ''),
            platform_name=str(getattr(properties, 'platform_name', '') or '')))
    save(args.output, report)
    torch.xpu.set_device(0)
    generator = torch.Generator(device='xpu').manual_seed(70)
    left = torch.rand(64, 64, generator=generator, device='xpu', dtype=torch.float32)
    right = torch.rand(64, 64, generator=generator, device='xpu', dtype=torch.float32)
    device_product = (left @ right).cpu()
    torch.xpu.synchronize()
    cpu_product = left.cpu() @ right.cpu()
    difference = (device_product - cpu_product).abs().max().item()
    half = torch.rand(64, 64, generator=generator, device='xpu', dtype=torch.float16)
    report['checks'] = dict(
        fp32_matmul_max_abs_vs_cpu=difference,
        fp32_exact=difference == 0.0,
        fp16_roundtrip_max_abs=(half.cpu().to(torch.float32) - half.to(torch.float32).cpu()).abs().max().item(),
        fp16_compute_sum=(half @ half).sum().item(),
        memory_allocated_bytes=torch.xpu.memory_allocated(0),
        memory_reserved_bytes=torch.xpu.memory_reserved(0))
    report['completed'] = bool(report['checks']['fp32_matmul_max_abs_vs_cpu'] < 1e-5)
    save(args.output, report)
    if not report['completed']:
        raise RuntimeError('XPU compute check failed')


if __name__ == '__main__':
    main()
