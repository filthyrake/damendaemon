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
    """MIG instances and slice geometry, or the honest unsupported reason.

    Per-GPU: a MIG-capable GPU reports its mode; a consumer card reports
    [N/A] for the same fields. The check is PER ROW — one consumer card
    next to a MIG-capable GPU must not hide the capable GPU's slices
    (the mixed A30+A2 host caught the whole-output 'N/A' shortcut lying).
    """
    binary = shutil.which('nvidia-smi')
    if binary is None:
        return dict(item='mig', passed=True, supported=False,
                    note='nvidia-smi unavailable; MIG inventory skipped')
    import subprocess
    query = subprocess.run([binary, '--query-gpu=index,name,mig.mode.current,mig.mode.pending',
                            '--format=csv'], capture_output=True, text=True, timeout=20)
    if query.returncode != 0:
        return dict(item='mig', passed=True, supported=False,
                    note='MIG not supported or not reportable on this GPU (consumer cards '
                         'report N/A); slice admission is an A30/data-center path')
    lines = [line for line in query.stdout.strip().splitlines() if line]
    rows = [dict(zip(('index', 'name', 'current', 'pending'),
                     (value.strip() for value in line.split(', '))))
            for line in lines[1:]]
    capable = [row for row in rows if row['current'] not in ('N/A', '[N/A]')]
    if not capable:
        return dict(item='mig', passed=True, supported=False,
                    note='no MIG-capable GPU on this host (consumer cards report '
                         'N/A); slice admission is an A30/data-center path')
    instances = []
    if any(row['current'] == 'Enabled' for row in capable):
        # nvidia-smi -L is the UNPRIVILEGED enumeration: it lists MIG device
        # UUIDs per GPU (the mig -lgi listing requires root and would turn a
        # read-only inventory into a permission failure).
        detail = subprocess.run([binary, '-L'], capture_output=True, text=True, timeout=20)
        instances = [line.strip() for line in detail.stdout.splitlines()
                     if 'MIG' in line]
    return dict(item='mig', passed=True,
                supported=any(row['current'] == 'Enabled' for row in capable),
                capable_gpus=capable, instances=instances,
                note='read-only; enabling MIG is a device-state change requiring a '
                     'maintenance window (#609 gate 3)')


def compute_mode_inventory():
    """Per-GPU compute mode: Exclusive/Prohibited modes block co-tenancy.

    nvidia-smi absence means the concept does not apply (non-NVIDIA host) or
    the tool is unavailable — either way this is a disclosed skip, not a
    failed preflight: MIG and MPS treat a missing nvidia-smi the same way.
    """
    rows = nvidia_smi('--query-gpu=index,name,compute_mode', '--format=csv')
    if rows is None or not rows:
        return dict(item='compute_mode', passed=True,
                    note='nvidia-smi unavailable; compute mode is an NVIDIA concept — '
                         'not applicable or unverifiable on this host')
    blocking = [row for row in rows if row['compute_mode'] not in ('Default', '[N/A]')]
    return dict(item='compute_mode', passed=True, gpus=rows,
                note='Exclusive/Prohibited modes block a second damendaemon process or MPS'
                     if blocking else 'default compute mode: co-tenancy is possible',
                blocking=blocking)


def mps_inventory():
    """MPS control binary, running server, and per-process view."""
    control = shutil.which('nvidia-cuda-mps-control')
    processes = nvidia_smi('--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv') or []
    mps_servers = [row for row in processes if 'mps' in row.get('process_name', '').lower()]
    return dict(item='mps', passed=True, control_binary=control,
                server_running=bool(mps_servers), server_processes=mps_servers,
                note=('control binary present; start nvidia-cuda-mps-control -d before the '
                      'co-tenant processes' if control else
                      'nvidia-cuda-mps-control absent: MPS unavailable on this host'))


def run_coTenancy_report():
    items = [mig_inventory(), compute_mode_inventory(), mps_inventory()]
    return dict(items=items, passed=all(item['passed'] for item in items))
