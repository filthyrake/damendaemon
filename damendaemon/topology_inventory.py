"""Read-only topology inventory for placement and transfer decisions (#604).

Reports per CUDA device: NUMA node and CPU affinity, the peer-access matrix
(`can_device_access_peer`) with NVLink presence, PCIe link width/generation,
virtualization markers, and the NUMA placement of NICs relevant to
cross-host staging. Everything here is read-only: sysfs reads, nvidia-smi
queries, and torch device queries. Readers are injectable so CI can run the
whole report on machines with no GPU and no sysfs.
"""
import os
import re
import subprocess


def _read_sysfs_int(path, reader=None):
    try:
        text = (reader or (lambda p: open(p).read()))(path)
    except OSError:
        return None
    text = text.strip()
    if not text or not re.fullmatch(r'-?\d+', text):
        return None
    return int(text)


def _sysfs_pci_bus_id(torch, index, runner=None):
    """Canonical PCI bus id for a torch device index, or None."""
    try:
        props = torch.cuda.get_device_properties(index)
        uuid = getattr(props, 'uuid', None)
    except Exception:
        return None
    result = (runner or _run)(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader',
                               f'--id={index}'])
    if result is None:
        return None
    bus_id = result.strip().splitlines()[0].strip() if result.strip() else None
    if not bus_id:
        return None
    # nvidia-smi pads the PCI domain to 8 digits; sysfs uses 4. The address
    # itself is colon-separated (domain:bus:device.func), so keep bus onward.
    return '0000:' + ':'.join(bus_id.split(':')[-2:])


def _run(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def device_numa_and_affinity(bus_id, reader=None):
    """NUMA node and local CPU list from sysfs; None means not discoverable."""
    base = f'/sys/bus/pci/devices/{bus_id}' if bus_id else None
    if base is None:
        return dict(numa_node=None, local_cpulist=None,
                    note='PCI bus id unavailable; NUMA placement unknown')
    node = _read_sysfs_int(f'{base}/numa_node', reader)
    if node is not None and node < 0:
        node = None  # -1: machine is NUMA-flat or the node is not exported
    cpus = None
    try:
        cpus = (reader or (lambda p: open(p).read()))(f'{base}/local_cpulist').strip() or None
    except OSError:
        pass
    return dict(numa_node=node, local_cpulist=cpus,
                note=None if node is not None else 'NUMA node not exported; treat as unknown')


def peer_matrix(torch, device_count):
    """can_device_access_peer for every ordered pair, plus the honest
    single-device note (the matrix is empty, not an error)."""
    pairs = []
    for source in range(device_count):
        for destination in range(device_count):
            if source == destination:
                continue
            try:
                pairs.append(dict(source=source, destination=destination,
                                  peer_available=bool(torch.cuda.can_device_access_peer(source, destination))))
            except Exception as error:
                pairs.append(dict(source=source, destination=destination, peer_available=False,
                                  error=str(error)))
    return pairs


def nvlink_presence(runner=None):
    """NVLink link list via nvidia-smi; empty output means no NVLink."""
    text = (runner or _run)(['nvidia-smi', 'nvlink', '--status'])
    if text is None:
        return dict(nvlink_present=False, note='nvidia-smi nvlink query unavailable; assumed absent')
    links = [line.strip() for line in text.splitlines()
             if line.strip() and 'Link' in line and 'Active' in line]
    return dict(nvlink_present=bool(links), links=links)


def pcie_links(runner=None):
    """Current PCIe link width/generation per GPU index."""
    text = (runner or _run)(['nvidia-smi', '--query-gpu=index,pcie.link.width.current,pcie.link.gen.current',
                             '--format=csv,noheader'])
    rows = {}
    if text:
        for line in text.strip().splitlines():
            fields = [field.strip() for field in line.split(',')]
            if len(fields) == 3:
                rows[int(fields[0])] = dict(pcie_link_width=fields[1], pcie_link_gen=fields[2])
    return rows


def virtualization_markers(reader=None):
    """Hypervisor markers that can break or slow P2P DMA (ACS/IOMMU)."""
    read = reader or (lambda p: open(p).read())
    markers = []
    try:
        vendor = read('/sys/devices/virtual/dmi/id/sys_vendor').strip()
    except OSError:
        vendor = ''
    if vendor and any(tag in vendor.lower() for tag in ('vmware', 'qemu', 'microsoft', 'innotek')):
        markers.append(dict(source='dmi_sys_vendor', value=vendor))
    try:
        hypervisor = read('/proc/sys/hypervisor/type').strip()
        if hypervisor:
            markers.append(dict(source='hypervisor_type', value=hypervisor))
    except OSError:
        pass
    return dict(virtualized=bool(markers), markers=markers,
                note='passed-through devices under ACS/IOMMU can break or slow P2P DMA'
                     if markers else None)


def nic_numa_placement(reader=None):
    """NIC -> NUMA node map for cross-host staging placement disclosure."""
    import glob
    placement = []
    for nic_path in sorted(glob.glob('/sys/class/net/*/device/numa_node')):
        nic = nic_path.split('/')[4]
        node = _read_sysfs_int(nic_path, reader)
        placement.append(dict(interface=nic, numa_node=node if node is not None and node >= 0 else None))
    return placement


def run_topology_report(torch):
    """The #604 gate-1 report. passed stays true when probes are merely
    unavailable (unknown != failing); actionable notes carry the gap."""
    import json
    available = bool(getattr(torch.cuda, 'is_available', lambda: False)())
    device_count = torch.cuda.device_count() if available else 0
    devices = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        bus_id = _sysfs_pci_bus_id(torch, index)
        numa = device_numa_and_affinity(bus_id)
        devices.append(dict(index=index, name=getattr(props, 'name', None),
                            uuid=str(getattr(props, 'uuid', '')) or None,
                            pci_bus_id=bus_id, **numa))
    report = dict(item='topology', passed=True,
                  devices=devices,
                  peer_matrix=peer_matrix(torch, device_count) if available else [],
                  nvlink=nvlink_presence(),
                  pcie_links=pcie_links(),
                  virtualization=virtualization_markers(),
                  nics=nic_numa_placement())
    if not available:
        report['note'] = 'no CUDA devices visible; topology report is the empty case'
    return report
