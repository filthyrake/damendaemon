"""Read-only NVIDIA co-tenancy inventory: MIG, compute mode, MPS presence.

Extends the fleet preflight per #609's inventory-first gate: enumerate MIG
instances with driver-reported slice geometry, compute mode, and MPS server
presence, with actionable errors when a query is unsupported. No device
state changes: every query here is read-only.
"""
import shutil
import subprocess


def nvidia_smi(*fields):
    """Run nvidia-smi query-gpu/-query-compute-apps; None when unavailable."""
    binary = shutil.which('nvidia-smi')
    if binary is None:
        return None
    import subprocess
    result = subprocess.run([binary, *fields], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.strip().splitlines() if line]
    if len(lines) < 2:
        return []
    header = [column.strip() for column in lines[0].split(', ')]
    return [dict(zip(header, (value.strip() for value in line.split(', '))))
            for line in lines[1:]]


def mig_inventory():
    """MIG instances and slice geometry, or the honest unsupported reason."""
    binary = shutil.which('nvidia-smi')
    if binary is None:
        return dict(item='mig', passed=True, supported=False,
                    note='nvidia-smi unavailable; MIG inventory skipped')
    import subprocess
    query = subprocess.run([binary, '--query-gpu=name,mig.mode.current,mig.mode.pending',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    if query.returncode != 0 or 'N/A' in query.stdout:
        return dict(item='mig', passed=True, supported=False,
                    note='MIG not supported or not reportable on this GPU (consumer cards '
                         'report N/A); slice admission is an A30/data-center path')
    supported = 'Enabled' in query.stdout
    instances = []
    if supported:
        detail = subprocess.run([binary, 'mig', '-l'], capture_output=True, text=True, timeout=20)
        instances = [line.strip() for line in detail.stdout.splitlines() if line.strip()]
    return dict(item='mig', passed=True, supported=supported,
                instances=instances,
                note='read-only; enabling MIG is a device-state change requiring a '
                     'maintenance window (#609 gate 3)')


def compute_mode_inventory():
    """Per-GPU compute mode: Exclusive/Prohibited modes block co-tenancy.

    nvidia-smi absence means the concept does not apply (non-NVIDIA host) or
    the tool is unavailable — either way this is a disclosed skip, not a
    failed preflight: MIG and MPS treat a missing nvidia-smi the same way.
    """
    rows = nvidia_smi('--query-gpu=index,name,compute_mode', '--format=csv,noheader')
    if rows is None:
        return dict(item='compute_mode', passed=True,
                    note='nvidia-smi unavailable; compute mode is an NVIDIA concept — '
                         'not applicable or unverifiable on this host')
    blocking = [row for row in rows if row['compute_mode'] not in ('Default',)]
    return dict(item='compute_mode', passed=True, gpus=rows,
                note='Exclusive/Prohibited modes block a second damendaemon process or MPS'
                     if blocking else 'default compute mode: co-tenancy is possible',
                blocking=blocking)


def mps_inventory():
    """MPS control binary, running server, and per-process view."""
    control = shutil.which('nvidia-cuda-mps-control')
    processes = nvidia_smi('--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader') or []
    mps_servers = [row for row in processes if 'mps' in row.get('process_name', '').lower()]
    return dict(item='mps', passed=True, control_binary=control,
                server_running=bool(mps_servers), server_processes=mps_servers,
                note=('control binary present; start nvidia-cuda-mps-control -d before the '
                      'co-tenant processes' if control else
                      'nvidia-cuda-mps-control absent: MPS unavailable on this host'))


def run_coTenancy_report():
    items = [mig_inventory(), compute_mode_inventory(), mps_inventory()]
    return dict(items=items, passed=all(item['passed'] for item in items))
