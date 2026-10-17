"""Bounded hybrid-placement controls: transfer bandwidth and projection costs.

Owned shared-run only; produces raw measured controls (host-to-device and
device-to-host pinned bandwidth, CPU packed matvec, GPU dense FP16 matvec) used
to choose explicit hybrid layer placement. Raw controls only — no placement
decision, no performance claim and no dispatch change is made here.
"""
import argparse
import json
import os
import time
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

TRANSFER_BYTES = (64*1024**2, 256*1024**2)
ROWS, COLUMNS = 4096, 11008  # representative packed FFN projection geometry


def inventory():
    import torch
    from damendaemon.rocm_policy import host_available
    model = ''
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            model = line.split(':', 1)[1].strip()
            break
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return dict(host_available_bytes=host_available(), host_total_bytes=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),
                cpu_count=os.cpu_count(), cpu_model=model, cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_name=props.name, gpu_total_bytes=total, gpu_free_bytes=free,
                cpu_threads_env=os.environ.get('OMP_NUM_THREADS'))


def _bandwidth(bytes_, direction, reps=5):
    import torch
    source = torch.empty(bytes_//4, dtype=torch.float32, pin_memory=True)
    device = torch.empty(bytes_//4, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        if direction == 'h2d':
            device.copy_(source, non_blocking=True)
        else:
            source.copy_(device, non_blocking=True)
        stop.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(stop)/1000
        best = seconds if best is None or seconds < best else best
    return dict(bytes=bytes_, direction=direction, best_seconds=best,
                bytes_per_second=bytes_/best, reps=reps)


def transfer_controls():
    return [_bandwidth(size, direction) for size in TRANSFER_BYTES for direction in ('h2d', 'd2h')]


def _q4_fixture(rows, columns):
    import torch
    blocks = columns//32
    codes = (torch.arange(rows*columns, dtype=torch.int64).reshape(rows, columns)*7+3) % 16
    groups = codes.reshape(rows, blocks, 32)
    weight = (groups[:, :, :16] | (groups[:, :, 16:] << 4)).to(torch.uint8).reshape(rows, columns//2)
    scales = ((torch.arange(rows*blocks).reshape(rows, blocks) % 7-3).to(torch.float16)*.125)
    return weight, scales


def projection_control(batch, reps=5):
    """CPU packed matvec with resident host weights; the streaming alternative
    is covered by the transfer controls and the GPU dense matvec ceiling."""
    import torch
    from damendaemon.packed_projection import packed_linear, projection_plan
    from damendaemon.packed_tiles import validate_layout
    weight, scales = _q4_fixture(ROWS, COLUMNS)
    validate_layout(weight, scales, format='q4_0_f16')
    projection_plan(weight, scales, format='q4_0_f16', row_chunk=64, column_chunk=256,
                    workspace_bytes=64*1024**2, device_type='cpu')
    hidden = torch.randn(batch, COLUMNS)
    value = packed_linear(hidden, weight, scales, format='q4_0_f16', row_chunk=64, column_chunk=256)  # warmup
    best = None
    for _ in range(reps):
        started = time.monotonic()
        value = packed_linear(hidden, weight, scales, format='q4_0_f16', row_chunk=64, column_chunk=256)
        seconds = time.monotonic()-started
        best = seconds if best is None or seconds < best else best
    if value.shape != (batch, ROWS) or not torch.isfinite(value).all():
        raise ValueError('CPU packed control produced invalid output')
    weight_bytes = weight.numel()+scales.numel()*2
    return dict(batch=batch, rows=ROWS, columns=COLUMNS, format='q4_0_f16',
                packed_bytes=weight_bytes, best_seconds=best, reps=reps)


def gpu_dense_control(batch, reps=5):
    """Dense FP16 matvec ceiling for the same geometry on device."""
    import torch
    weight = torch.randn(ROWS, COLUMNS, dtype=torch.float16, device='cuda')
    hidden = torch.randn(batch, COLUMNS, dtype=torch.float16, device='cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    for _ in range(2):
        torch.matmul(hidden, weight.T)
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        torch.matmul(hidden, weight.T)
        stop.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(stop)/1000
        best = seconds if best is None or seconds < best else best
    return dict(batch=batch, rows=ROWS, columns=COLUMNS, dtype='float16',
                best_seconds=best, reps=reps, weight_bytes=weight.numel()*2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report = dict(completed=False, torch=torch.__version__,
                  device=torch.cuda.get_device_name(), rows=ROWS, columns=COLUMNS,
                  transfer_bytes=list(TRANSFER_BYTES), controls={})
    save(args.output, report)
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS') or os.cpu_count()))
        report['inventory'] = inventory()
        report['controls']['transfers'] = transfer_controls()
        report['controls']['cpu_packed'] = [projection_control(batch) for batch in (1, 4)]
        report['controls']['gpu_dense'] = [gpu_dense_control(batch) for batch in (1, 4)]
        report['completed'] = True
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
